// RDMA transport for ggml-rpc — Raw Verbs edition
// Bypasses RDMA CM (which fails with RoCE v1 on ConnectX-3)
// Uses a small TCP socket for QP info exchange, then RDMA verbs for data
#pragma once

#include <cinttypes>
#include <infiniband/verbs.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <algorithm>
#include <arpa/inet.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <unistd.h>
#include <errno.h>

static constexpr size_t RDMA_BUF_SIZE = 64 * 1024 * 1024; // 64 MiB
static constexpr int    RDMA_MAX_WR   = 32;
static constexpr int    RDMA_MAX_SGE  = 1;
static constexpr int    RDMA_GID_IDX  = 1; // RoCE v1 IPv4 GID

// QP info exchanged over TCP for connection setup
struct qp_info_t {
    uint32_t qpn;
    uint32_t psn;
    union ibv_gid gid;
};

// Simple TCP helpers for QP info exchange only
static bool tcp_send_all(int fd, const void *buf, size_t len) {
    const char *p = (const char *)buf;
    while (len > 0) {
        ssize_t n = send(fd, p, len, 0);
        if (n <= 0) return false;
        p += n; len -= n;
    }
    return true;
}

static bool tcp_recv_all(int fd, void *buf, size_t len) {
    char *p = (char *)buf;
    while (len > 0) {
        ssize_t n = recv(fd, p, len, 0);
        if (n <= 0) return false;
        p += n; len -= n;
    }
    return true;
}

struct rdma_transport_t {
    struct ibv_context    *ctx     = nullptr;
    struct ibv_pd         *pd      = nullptr;
    struct ibv_cq         *send_cq = nullptr;  // separate CQ for send completions
    struct ibv_cq         *recv_cq = nullptr;  // separate CQ for recv completions
    struct ibv_qp         *qp      = nullptr;
    struct ibv_mr         *send_mr = nullptr;
    struct ibv_mr         *recv_mr = nullptr;
    uint8_t               *send_buf = nullptr;
    uint8_t               *recv_buf = nullptr;
    bool                   connected = false;
    int                    tcp_fd   = -1;       // TCP socket for control (QP exchange)
    int                    listen_fd = -1;      // TCP listen socket (server only)

    ~rdma_transport_t() {
        if (qp)      ibv_destroy_qp(qp);
        if (send_mr) ibv_dereg_mr(send_mr);
        if (recv_mr) ibv_dereg_mr(recv_mr);
        if (send_cq) ibv_destroy_cq(send_cq);
        if (recv_cq) ibv_destroy_cq(recv_cq);
        if (pd)      ibv_dealloc_pd(pd);
        if (ctx)     ibv_close_device(ctx);
        if (send_buf) free(send_buf);
        if (recv_buf) free(recv_buf);
        if (tcp_fd >= 0) close(tcp_fd);
        if (listen_fd >= 0) close(listen_fd);
    }

    bool open_device() {
        int num_devices;
        struct ibv_device **dev_list = ibv_get_device_list(&num_devices);
        if (!dev_list || num_devices == 0) {
            fprintf(stderr, "RDMA: no devices found\n");
            return false;
        }
        ctx = ibv_open_device(dev_list[0]);
        ibv_free_device_list(dev_list);
        if (!ctx) {
            fprintf(stderr, "RDMA: failed to open device\n");
            return false;
        }
        return true;
    }

    bool alloc_buffers() {
        send_buf = (uint8_t *)aligned_alloc(4096, RDMA_BUF_SIZE);
        recv_buf = (uint8_t *)aligned_alloc(4096, RDMA_BUF_SIZE);
        if (!send_buf || !recv_buf) {
            fprintf(stderr, "RDMA: buffer alloc failed\n");
            return false;
        }
        memset(send_buf, 0, RDMA_BUF_SIZE);
        memset(recv_buf, 0, RDMA_BUF_SIZE);
        return true;
    }

    bool setup_resources() {
        pd = ibv_alloc_pd(ctx);
        if (!pd) { fprintf(stderr, "RDMA: PD alloc failed\n"); return false; }

        // Separate CQs so send_data and recv_data never steal each other's completions
        send_cq = ibv_create_cq(ctx, RDMA_MAX_WR, nullptr, nullptr, 0);
        recv_cq = ibv_create_cq(ctx, RDMA_MAX_WR, nullptr, nullptr, 0);
        if (!send_cq || !recv_cq) { fprintf(stderr, "RDMA: CQ create failed\n"); return false; }

        send_mr = ibv_reg_mr(pd, send_buf, RDMA_BUF_SIZE, IBV_ACCESS_LOCAL_WRITE);
        recv_mr = ibv_reg_mr(pd, recv_buf, RDMA_BUF_SIZE,
                             IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
        if (!send_mr || !recv_mr) {
            fprintf(stderr, "RDMA: MR reg failed\n");
            return false;
        }

        // Create QP with separate send/recv CQs
        struct ibv_qp_init_attr qp_init = {};
        qp_init.send_cq = send_cq;
        qp_init.recv_cq = recv_cq;
        qp_init.cap.max_send_wr  = RDMA_MAX_WR;
        qp_init.cap.max_recv_wr  = RDMA_MAX_WR;
        qp_init.cap.max_send_sge = RDMA_MAX_SGE;
        qp_init.cap.max_recv_sge = RDMA_MAX_SGE;
        qp_init.qp_type = IBV_QPT_RC;
        qp_init.sq_sig_all = 1;

        qp = ibv_create_qp(pd, &qp_init);
        if (!qp) {
            fprintf(stderr, "RDMA: QP create failed: %s\n", strerror(errno));
            return false;
        }
        return true;
    }

    bool get_local_info(qp_info_t &info) {
        info.qpn = qp->qp_num;
        info.psn = rand() & 0xFFFFFF;
        if (ibv_query_gid(ctx, 1, RDMA_GID_IDX, &info.gid)) {
            fprintf(stderr, "RDMA: query GID failed\n");
            return false;
        }
        return true;
    }

    bool modify_qp_to_init() {
        struct ibv_qp_attr attr = {};
        attr.qp_state        = IBV_QPS_INIT;
        attr.pkey_index      = 0;
        attr.port_num        = 1;
        attr.qp_access_flags = IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_LOCAL_WRITE;
        if (ibv_modify_qp(qp, &attr,
                          IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS)) {
            fprintf(stderr, "RDMA: QP->INIT failed: %s\n", strerror(errno));
            return false;
        }
        return true;
    }

    bool modify_qp_to_rtr(const qp_info_t &remote) {
        struct ibv_qp_attr attr = {};
        attr.qp_state              = IBV_QPS_RTR;
        attr.path_mtu              = IBV_MTU_1024;
        attr.dest_qp_num           = remote.qpn;
        attr.rq_psn                = remote.psn;
        attr.max_dest_rd_atomic    = 1;
        attr.min_rnr_timer         = 12;
        attr.ah_attr.dlid          = 0;
        attr.ah_attr.sl            = 0;
        attr.ah_attr.src_path_bits = 0;
        attr.ah_attr.port_num      = 1;
        attr.ah_attr.is_global     = 1;
        attr.ah_attr.grh.dgid      = remote.gid;
        attr.ah_attr.grh.sgid_index = RDMA_GID_IDX;
        attr.ah_attr.grh.hop_limit  = 1;
        attr.ah_attr.grh.flow_label = 0;
        attr.ah_attr.grh.traffic_class = 0;

        if (ibv_modify_qp(qp, &attr,
                          IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU |
                          IBV_QP_DEST_QPN | IBV_QP_RQ_PSN |
                          IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER)) {
            fprintf(stderr, "RDMA: QP->RTR failed: %s\n", strerror(errno));
            return false;
        }
        return true;
    }

    bool modify_qp_to_rts(const qp_info_t &local) {
        struct ibv_qp_attr attr = {};
        attr.qp_state      = IBV_QPS_RTS;
        attr.timeout        = 14;
        attr.retry_cnt      = 7;
        attr.rnr_retry      = 7;
        attr.sq_psn         = local.psn;
        attr.max_rd_atomic  = 1;

        if (ibv_modify_qp(qp, &attr,
                          IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT |
                          IBV_QP_RNR_RETRY | IBV_QP_SQ_PSN | IBV_QP_MAX_QP_RD_ATOMIC)) {
            fprintf(stderr, "RDMA: QP->RTS failed: %s\n", strerror(errno));
            return false;
        }
        return true;
    }

    bool exchange_and_connect(bool is_server) {
        qp_info_t local_info, remote_info;
        if (!get_local_info(local_info)) return false;
        if (!modify_qp_to_init()) return false;

        if (is_server) {
            if (!tcp_recv_all(tcp_fd, &remote_info, sizeof(remote_info))) {
                fprintf(stderr, "RDMA: failed to recv remote QP info\n");
                return false;
            }
            if (!tcp_send_all(tcp_fd, &local_info, sizeof(local_info))) {
                fprintf(stderr, "RDMA: failed to send local QP info\n");
                return false;
            }
        } else {
            if (!tcp_send_all(tcp_fd, &local_info, sizeof(local_info))) {
                fprintf(stderr, "RDMA: failed to send local QP info\n");
                return false;
            }
            if (!tcp_recv_all(tcp_fd, &remote_info, sizeof(remote_info))) {
                fprintf(stderr, "RDMA: failed to recv remote QP info\n");
                return false;
            }
        }

        fprintf(stderr, "RDMA: local  QPN=0x%06x PSN=0x%06x\n", local_info.qpn, local_info.psn);
        fprintf(stderr, "RDMA: remote QPN=0x%06x PSN=0x%06x\n", remote_info.qpn, remote_info.psn);

        if (!modify_qp_to_rtr(remote_info)) return false;
        if (!modify_qp_to_rts(local_info)) return false;

        // Pre-post a receive so we're ready for the first incoming message
        post_recv(RDMA_BUF_SIZE);

        connected = true;
        fprintf(stderr, "RDMA: QP connected!\n");
        return true;
    }

    bool post_recv(size_t size) {
        struct ibv_sge sge = {};
        sge.addr   = (uint64_t)recv_buf;
        sge.length = std::min(size, RDMA_BUF_SIZE);
        sge.lkey   = recv_mr->lkey;
        struct ibv_recv_wr wr = {}, *bad = nullptr;
        wr.sg_list = &sge;
        wr.num_sge = 1;
        if (ibv_post_recv(qp, &wr, &bad)) {
            fprintf(stderr, "RDMA: post_recv failed\n");
            return false;
        }
        return true;
    }

    // Poll send CQ only — never steals recv completions
    bool poll_send_cq() {
        struct ibv_wc wc;
        while (ibv_poll_cq(send_cq, 1, &wc) == 0) { /* busy-poll */ }
        if (wc.status != IBV_WC_SUCCESS) {
            fprintf(stderr, "RDMA: send CQ error: status=%d (%s) opcode=%d\n",
                    wc.status, ibv_wc_status_str(wc.status), wc.opcode);
            return false;
        }
        return true;
    }

    // Poll recv CQ only — never steals send completions
    bool poll_recv_cq() {
        struct ibv_wc wc;
        while (ibv_poll_cq(recv_cq, 1, &wc) == 0) { /* busy-poll */ }
        if (wc.status != IBV_WC_SUCCESS) {
            fprintf(stderr, "RDMA: recv CQ error: status=%d (%s) opcode=%d\n",
                    wc.status, ibv_wc_status_str(wc.status), wc.opcode);
            return false;
        }
        return true;
    }

    bool send_data(const void *data, size_t size) {
        const uint8_t *src = (const uint8_t *)data;
        size_t offset = 0;
        while (offset < size) {
            size_t chunk = std::min(size - offset, RDMA_BUF_SIZE - 8);
            uint64_t len = chunk;
            memcpy(send_buf, &len, 8);
            memcpy(send_buf + 8, src + offset, chunk);

            struct ibv_sge sge = {};
            sge.addr   = (uint64_t)send_buf;
            sge.length = chunk + 8;
            sge.lkey   = send_mr->lkey;

            struct ibv_send_wr wr = {}, *bad = nullptr;
            wr.sg_list    = &sge;
            wr.num_sge    = 1;
            wr.opcode     = IBV_WR_SEND;
            wr.send_flags = IBV_SEND_SIGNALED;
            if (chunk + 8 <= 236) wr.send_flags |= IBV_SEND_INLINE;

            if (ibv_post_send(qp, &wr, &bad)) {
                fprintf(stderr, "RDMA: post_send failed\n");
                return false;
            }
            if (!poll_send_cq()) return false;
            offset += chunk;
        }
        return true;
    }

    bool recv_data(void *data, size_t size) {
        uint8_t *dst = (uint8_t *)data;
        size_t offset = 0;
        while (offset < size) {
            if (!poll_recv_cq()) return false;
            uint64_t chunk_len;
            memcpy(&chunk_len, recv_buf, 8);
            if (chunk_len > RDMA_BUF_SIZE - 8 || chunk_len > size - offset) {
                fprintf(stderr, "RDMA: bad chunk len %zu (remaining %zu)\n",
                        (size_t)chunk_len, size - offset);
                return false;
            }
            memcpy(dst + offset, recv_buf + 8, chunk_len);
            offset += chunk_len;
            if (!post_recv(RDMA_BUF_SIZE)) return false;
        }
        return true;
    }
};

// ---- Connection setup functions ----
#include <map>
#include <mutex>

static std::mutex rdma_cache_mutex;
static std::map<std::string, std::shared_ptr<rdma_transport_t>> rdma_cache;

static std::shared_ptr<rdma_transport_t> rdma_transport_connect(const char *host, int port) {
    auto t = std::make_shared<rdma_transport_t>();

    if (!t->open_device()) return nullptr;
    if (!t->alloc_buffers()) return nullptr;
    if (!t->setup_resources()) return nullptr;

    t->tcp_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (t->tcp_fd < 0) {
        fprintf(stderr, "RDMA: TCP socket failed\n");
        return nullptr;
    }
    int flag = 1;
    setsockopt(t->tcp_fd, IPPROTO_TCP, TCP_NODELAY, &flag, sizeof(flag));

    struct sockaddr_in addr = {};
    addr.sin_family = AF_INET;
    addr.sin_port   = htons(port + 1);
    inet_pton(AF_INET, host, &addr.sin_addr);

    if (connect(t->tcp_fd, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
        fprintf(stderr, "RDMA: TCP connect to %s:%d failed: %s\n", host, port + 1, strerror(errno));
        return nullptr;
    }
    fprintf(stderr, "RDMA: TCP control connected to %s:%d\n", host, port + 1);

    if (!t->exchange_and_connect(false)) return nullptr;

    return t;
}

static std::shared_ptr<rdma_transport_t> rdma_transport_connect_cached(const char *host, int port) {
    char key[256];
    snprintf(key, sizeof(key), "%s:%d", host, port);
    fprintf(stderr, "RDMA: cache lookup key='%s'\n", key);
    std::lock_guard<std::mutex> lock(rdma_cache_mutex);
    auto it = rdma_cache.find(key);
    if (it != rdma_cache.end() && it->second && it->second->connected) {
        fprintf(stderr, "RDMA: cache HIT\n");
        return it->second;
    }
    fprintf(stderr, "RDMA: cache MISS\n");   
    auto t = rdma_transport_connect(host, port);
    if (t) {
        rdma_cache[key] = t;
    }
    return t;
}

static std::shared_ptr<rdma_transport_t> rdma_transport_listen(const char *host, int port) {
    auto t = std::make_shared<rdma_transport_t>();

    t->listen_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (t->listen_fd < 0) {
        fprintf(stderr, "RDMA: TCP listen socket failed\n");
        return nullptr;
    }
    int flag = 1;
    setsockopt(t->listen_fd, SOL_SOCKET, SO_REUSEADDR, &flag, sizeof(flag));

    struct sockaddr_in addr = {};
    addr.sin_family = AF_INET;
    addr.sin_port   = htons(port + 1);
    if (host && strlen(host) > 0 && strcmp(host, "0.0.0.0") != 0)
        inet_pton(AF_INET, host, &addr.sin_addr);
    else
        addr.sin_addr.s_addr = INADDR_ANY;

    if (bind(t->listen_fd, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
        fprintf(stderr, "RDMA: TCP bind failed on port %d: %s\n", port + 1, strerror(errno));
        return nullptr;
    }
    if (listen(t->listen_fd, 1) < 0) {
        fprintf(stderr, "RDMA: TCP listen failed\n");
        return nullptr;
    }
    fprintf(stderr, "RDMA: listening on %s:%d (TCP control on port %d)\n",
            host ? host : "0.0.0.0", port, port + 1);
    return t;
}

static std::shared_ptr<rdma_transport_t> rdma_transport_accept(std::shared_ptr<rdma_transport_t> srv) {
    auto t = std::make_shared<rdma_transport_t>();

    t->tcp_fd = accept(srv->listen_fd, nullptr, nullptr);
    if (t->tcp_fd < 0) {
        fprintf(stderr, "RDMA: TCP accept failed\n");
        return nullptr;
    }
    int flag = 1;
    setsockopt(t->tcp_fd, IPPROTO_TCP, TCP_NODELAY, &flag, sizeof(flag));
    fprintf(stderr, "RDMA: TCP control client connected\n");

    if (!t->open_device()) return nullptr;
    if (!t->alloc_buffers()) return nullptr;
    if (!t->setup_resources()) return nullptr;

    if (!t->exchange_and_connect(true)) return nullptr;

    return t;
}

static bool is_rdma_enabled() {
    static int cached = -1;
    if (cached < 0) {
        const char *env = std::getenv("GGML_RPC_RDMA");
        cached = (env && (strcmp(env, "1") == 0 || strcmp(env, "true") == 0)) ? 1 : 0;
        if (cached) fprintf(stderr, "RDMA: transport enabled\n");
    }
    return cached == 1;
}

