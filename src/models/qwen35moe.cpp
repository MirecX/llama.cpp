#include "models.h"

#include "llama-memory-recurrent.h"
#include "ggml-backend.h"

llm_build_qwen35moe::llm_build_qwen35moe(const llama_model & model, const llm_graph_params & params) :
    llm_build_delta_net_base(params), model(model) {
    const int64_t n_embd_head = hparams.n_embd_head_v;

    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k);

    int sections[4];
    std::copy(std::begin(hparams.rope_sections), std::begin(hparams.rope_sections) + 4, sections);

    ggml_tensor * cur;
    ggml_tensor * inpL;

    inpL = build_inp_embd(model.tok_embd);

    cb(inpL, "model.input_embed", -1);

    auto * inp = build_inp_mem_hybrid();

    ggml_tensor * inp_pos     = build_inp_pos();
    ggml_tensor * inp_out_ids = build_inp_out_ids();

    for (int il = 0; il < n_layer; ++il) {
        ggml_tensor * inpSA = inpL;

        cur = build_norm(inpL, model.layers[il].attn_norm, nullptr, LLM_NORM_RMS, il);
        cb(cur, "attn_norm", il);

        // Determine layer type and build appropriate attention mechanism
        if (hparams.is_recurrent(il)) {
            // Linear attention layer (gated delta net)
            cur = build_layer_attn_linear(inp->get_recr(), cur, il);
        } else {
            // Full attention layer
            cur = build_layer_attn(inp->get_attn(), cur, inp_pos, sections, il);
        }

        if (il == n_layer - 1 && inp_out_ids) {
            cur   = ggml_get_rows(ctx0, cur, inp_out_ids);
            inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
        }

        // Residual connection
        cur = ggml_add(ctx0, cur, inpSA);
        cb(cur, "attn_residual", il);

        // Save the tensor before post-attention norm for residual connection
        ggml_tensor * ffn_residual = cur;

        // Post-attention norm
        ggml_tensor * attn_post_norm = build_norm(cur, model.layers[il].attn_post_norm, nullptr, LLM_NORM_RMS, il);
        cb(attn_post_norm, "attn_post_norm", il);

        // When TP is active, pin attention/residual/norm ops to primary device.
        // This prevents the scheduler from absorbing attention-related view ops
        // (e.g., conv_states RESHAPE/VIEW) into RPC splits via pass 5 expansion.
        // View ops don't get their own split boundaries in the scheduler, so without
        // these pins, state cache ops with Vulkan buffers can end up in RPC splits.
        if (tp) {
            ggml_backend_t primary = get_tp_backend(0);
            ggml_backend_sched_set_tensor_backend(sched, cur, primary);
            ggml_backend_sched_set_tensor_backend(sched, ffn_residual, primary);
            ggml_backend_sched_set_tensor_backend(sched, attn_post_norm, primary);
        }

        // MOE FFN layer
        cur = build_layer_ffn(attn_post_norm, il);
        cb(cur, "ffn_out", il);

        // Residual connection for FFN - add to the tensor from before post_attention_layernorm
        cur = ggml_add(ctx0, cur, ffn_residual);
        cb(cur, "post_moe", il);

        // When TP active, pin final residual to primary
        if (tp) {
            ggml_backend_sched_set_tensor_backend(sched, cur, get_tp_backend(0));
        }

        // Input for next layer
        inpL = cur;
    }
    cur = inpL;

    // Final norm
    cur = build_norm(cur, model.output_norm, nullptr, LLM_NORM_RMS, -1);

    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    // LM head
    cur = build_lora_mm(model.output, cur);

    cb(cur, "result_output", -1);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
}

std::pair<ggml_tensor *, ggml_tensor *> llm_build_qwen35moe::build_qkvz(
                ggml_tensor * input,
                        int   il) {
    const int64_t n_seqs       = ubatch.n_seqs;
    const int64_t n_seq_tokens = ubatch.n_seq_tokens;

    ggml_tensor * qkv_mixed = build_lora_mm(model.layers[il].wqkv, input);
    qkv_mixed = ggml_reshape_3d(ctx0, qkv_mixed, qkv_mixed->ne[0], n_seq_tokens, n_seqs);
    cb(qkv_mixed, "linear_attn_qkv_mixed", il);

    ggml_tensor * z = build_lora_mm(model.layers[il].wqkv_gate, input);
    cb(z, "z", il);

    return { qkv_mixed, z };
}

ggml_tensor * llm_build_qwen35moe::build_norm_gated(
        ggml_tensor * input,
        ggml_tensor * weights,
        ggml_tensor * gate,
        int           layer) {
    ggml_tensor * normalized = build_norm(input, weights, nullptr, LLM_NORM_RMS, layer);
    ggml_tensor * gated_silu = ggml_silu(ctx0, gate);

    return ggml_mul(ctx0, normalized, gated_silu);
}

ggml_tensor * llm_build_qwen35moe ::build_layer_attn(
        llm_graph_input_attn_kv * inp,
        ggml_tensor *             cur,
        ggml_tensor *             inp_pos,
        int *                     sections,
        int                       il) {
    const int64_t n_embd_head = hparams.n_embd_head_v;
    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k);

    // TP for attention is disabled for now: build_attn manages the KV cache
    // internally and expects full head dimensions. Only FFN uses TP.
    // TODO: implement TP-aware KV cache splitting

    // Order: joint QG projection, QG split, Q norm, KV projection, K norm, RoPE, attention

    // Qwen3Next uses a single Q projection that outputs query + gate
    ggml_tensor * Qcur_full = build_lora_mm(model.layers[il].wq, cur); // [ (n_embd_head * 2) * n_head, n_tokens ]
    cb(Qcur_full, "Qcur_full", il);

    ggml_tensor * Qcur = ggml_view_3d(ctx0, Qcur_full, n_embd_head, n_head, n_tokens,
        ggml_element_size(Qcur_full) * n_embd_head * 2,
        ggml_element_size(Qcur_full) * n_embd_head * 2 * n_head, 0);
    cb(Qcur, "Qcur_reshaped", il);

    // Apply Q normalization
    Qcur = build_norm(Qcur, model.layers[il].attn_q_norm, nullptr, LLM_NORM_RMS, il);
    cb(Qcur, "Qcur_normed", il);

    ggml_tensor * Kcur = build_lora_mm(model.layers[il].wk, cur);
    cb(Kcur, "Kcur", il);

    ggml_tensor * Vcur = build_lora_mm(model.layers[il].wv, cur);
    cb(Vcur, "Vcur", il);

    // Apply K normalization
    Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);
    Kcur = build_norm(Kcur, model.layers[il].attn_k_norm, nullptr, LLM_NORM_RMS, il);
    cb(Kcur, "Kcur_normed", il);

    ggml_tensor * gate = ggml_view_3d(ctx0, Qcur_full, n_embd_head, n_head, n_tokens,
        ggml_element_size(Qcur_full) * n_embd_head * 2,
        ggml_element_size(Qcur_full) * n_embd_head * 2 * n_head,
        ggml_element_size(Qcur_full) * n_embd_head);
    gate = ggml_cont_2d(ctx0, gate, n_embd_head * n_head, n_tokens);
    cb(gate, "gate_reshaped", il);

    Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_tokens);

    // Apply IMRoPE
    Qcur = ggml_rope_multi(
            ctx0, Qcur, inp_pos, nullptr,
            n_rot, sections, rope_type, n_ctx_orig, freq_base, freq_scale,
            ext_factor, attn_factor, beta_fast, beta_slow
            );

    Kcur = ggml_rope_multi(
            ctx0, Kcur, inp_pos, nullptr,
            n_rot, sections, rope_type, n_ctx_orig, freq_base, freq_scale,
            ext_factor, attn_factor, beta_fast, beta_slow
            );

    cb(Qcur, "Qcur", il);
    cb(Kcur, "Kcur", il);
    cb(Vcur, "Vcur", il);

    // Attention computation
    const float kq_scale = hparams.f_attention_scale == 0.0f ? 1.0f / sqrtf(float(n_embd_head)) : hparams.f_attention_scale;

    cur = build_attn(inp,
                nullptr, nullptr,
                Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, kq_scale, il);
    cb(cur, "attn_pregate", il);

    ggml_tensor * gate_sigmoid = ggml_sigmoid(ctx0, gate);
    cb(gate_sigmoid, "gate_sigmoid", il);

    cur = ggml_mul(ctx0, cur, gate_sigmoid);
    cb(cur, "attn_gated", il);

    cur = build_lora_mm(model.layers[il].wo, cur);
    cb(cur, "attn_output", il);

    return cur;
}

ggml_tensor * llm_build_qwen35moe ::build_layer_attn_tp(
        llm_graph_input_attn_kv * inp,
        ggml_tensor *             cur,
        ggml_tensor *             inp_pos,
        int *                     sections,
        int                       il) {
    // TP attention for Qwen3.5 MoE standard attention layers
    // Handles the fused Q+gate projection by splitting along the head dimension
    const int64_t n_embd_head = hparams.n_embd_head_v;
    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k);
    GGML_ASSERT(n_head % 2 == 0);
    GGML_ASSERT(n_head_kv % 2 == 0);

    const int64_t half_n_head    = n_head / 2;
    const int64_t half_n_head_kv = n_head_kv / 2;

    // Column-parallel Q+gate projection: wq has shape [n_embd, n_embd_head*2*n_head]
    // Split along ne[1] gives each device n_embd_head*2*n_head/2 = n_embd_head*2*half_n_head outputs
    auto [qg0, qg1] = build_lora_mm_tp_col(model.layers[il].wq, cur);
    cb(qg0, "Qcur_full_tp0", il);
    cb(qg1, "Qcur_full_tp1", il);

    // Extract Q and gate from each shard (each has half_n_head heads with 2*n_embd_head per head)
    // Q: stride-2 view extracting first n_embd_head of each 2*n_embd_head block
    ggml_tensor * Qcur0 = ggml_view_3d(ctx0, qg0, n_embd_head, half_n_head, n_tokens,
        ggml_element_size(qg0) * n_embd_head * 2,
        ggml_element_size(qg0) * n_embd_head * 2 * half_n_head, 0);
    ggml_tensor * Qcur1 = ggml_view_3d(ctx0, qg1, n_embd_head, half_n_head, n_tokens,
        ggml_element_size(qg1) * n_embd_head * 2,
        ggml_element_size(qg1) * n_embd_head * 2 * half_n_head, 0);

    // Apply Q normalization (per-head, so works on each shard independently)
    Qcur0 = build_norm(Qcur0, model.layers[il].attn_q_norm, nullptr, LLM_NORM_RMS, il);
    Qcur1 = build_norm(Qcur1, model.layers[il].attn_q_norm, nullptr, LLM_NORM_RMS, il);

    // Gate: extract second n_embd_head of each 2*n_embd_head block
    ggml_tensor * gate0 = ggml_view_3d(ctx0, qg0, n_embd_head, half_n_head, n_tokens,
        ggml_element_size(qg0) * n_embd_head * 2,
        ggml_element_size(qg0) * n_embd_head * 2 * half_n_head,
        ggml_element_size(qg0) * n_embd_head);
    ggml_tensor * gate1 = ggml_view_3d(ctx0, qg1, n_embd_head, half_n_head, n_tokens,
        ggml_element_size(qg1) * n_embd_head * 2,
        ggml_element_size(qg1) * n_embd_head * 2 * half_n_head,
        ggml_element_size(qg1) * n_embd_head);
    gate0 = ggml_cont_2d(ctx0, gate0, n_embd_head * half_n_head, n_tokens);
    gate1 = ggml_cont_2d(ctx0, gate1, n_embd_head * half_n_head, n_tokens);

    // Column-parallel K and V projections
    auto [k0, k1] = build_lora_mm_tp_col(model.layers[il].wk, cur);
    auto [v0, v1] = build_lora_mm_tp_col(model.layers[il].wv, cur);

    // Reshape K and V to [n_embd_head, half_n_head_kv, n_tokens]
    ggml_tensor * Kcur0 = ggml_reshape_3d(ctx0, k0, n_embd_head, half_n_head_kv, n_tokens);
    ggml_tensor * Kcur1 = ggml_reshape_3d(ctx0, k1, n_embd_head, half_n_head_kv, n_tokens);
    ggml_tensor * Vcur0 = ggml_reshape_3d(ctx0, v0, n_embd_head, half_n_head_kv, n_tokens);
    ggml_tensor * Vcur1 = ggml_reshape_3d(ctx0, v1, n_embd_head, half_n_head_kv, n_tokens);

    // Apply K normalization (per-head)
    Kcur0 = build_norm(Kcur0, model.layers[il].attn_k_norm, nullptr, LLM_NORM_RMS, il);
    Kcur1 = build_norm(Kcur1, model.layers[il].attn_k_norm, nullptr, LLM_NORM_RMS, il);

    // Apply IMRoPE to each shard
    Qcur0 = ggml_rope_multi(ctx0, Qcur0, inp_pos, nullptr,
            n_rot, sections, rope_type, n_ctx_orig, freq_base, freq_scale,
            ext_factor, attn_factor, beta_fast, beta_slow);
    Qcur1 = ggml_rope_multi(ctx0, Qcur1, inp_pos, nullptr,
            n_rot, sections, rope_type, n_ctx_orig, freq_base, freq_scale,
            ext_factor, attn_factor, beta_fast, beta_slow);
    Kcur0 = ggml_rope_multi(ctx0, Kcur0, inp_pos, nullptr,
            n_rot, sections, rope_type, n_ctx_orig, freq_base, freq_scale,
            ext_factor, attn_factor, beta_fast, beta_slow);
    Kcur1 = ggml_rope_multi(ctx0, Kcur1, inp_pos, nullptr,
            n_rot, sections, rope_type, n_ctx_orig, freq_base, freq_scale,
            ext_factor, attn_factor, beta_fast, beta_slow);

    cb(Qcur0, "Qcur_tp0", il);
    cb(Qcur1, "Qcur_tp1", il);
    cb(Kcur0, "Kcur_tp0", il);
    cb(Kcur1, "Kcur_tp1", il);

    // Attention for each shard (each device handles its half of heads independently)
    const float kq_scale = hparams.f_attention_scale == 0.0f ? 1.0f / sqrtf(float(n_embd_head)) : hparams.f_attention_scale;

    // Note: passing wo=nullptr so build_attn doesn't apply output projection
    ggml_tensor * attn0 = build_attn(inp, nullptr, nullptr, Qcur0, Kcur0, Vcur0, nullptr, nullptr, nullptr, kq_scale, il);
    ggml_tensor * attn1 = build_attn(inp, nullptr, nullptr, Qcur1, Kcur1, Vcur1, nullptr, nullptr, nullptr, kq_scale, il);
    cb(attn0, "attn_tp0", il);
    cb(attn1, "attn_tp1", il);

    // Apply gate sigmoid to each shard
    ggml_tensor * gate_sigmoid0 = ggml_sigmoid(ctx0, gate0);
    ggml_tensor * gate_sigmoid1 = ggml_sigmoid(ctx0, gate1);
    attn0 = ggml_mul(ctx0, attn0, gate_sigmoid0);
    attn1 = ggml_mul(ctx0, attn1, gate_sigmoid1);
    cb(attn0, "attn_gated_tp0", il);
    cb(attn1, "attn_gated_tp1", il);

    // Hint backends
    ggml_backend_sched_set_tensor_backend(sched, attn0, get_tp_backend(0));
    ggml_backend_sched_set_tensor_backend(sched, attn1, get_tp_backend(1));

    // Row-parallel output projection + all-reduce
    cur = build_lora_mm_tp_row(model.layers[il].wo, attn0, attn1);
    cb(cur, "attn_output_tp", il);

    return cur;
}

ggml_tensor * llm_build_qwen35moe ::build_layer_attn_linear(
        llm_graph_input_rs * inp,
        ggml_tensor *        cur,
        int                  il) {
    const auto * mctx_cur = inp->mctx;

    const int64_t d_inner      = hparams.ssm_d_inner;
    const int64_t n_seqs       = ubatch.n_seqs;
    const int64_t head_k_dim   = hparams.ssm_d_state;
    const int64_t num_k_heads  = hparams.ssm_n_group;
    const int64_t num_v_heads  = hparams.ssm_dt_rank;
    const int64_t head_v_dim   = d_inner / num_v_heads;
    const int64_t n_seq_tokens = ubatch.n_seq_tokens;

    const auto kv_head = mctx_cur->get_head();

    GGML_ASSERT(n_seqs != 0);
    GGML_ASSERT(ubatch.equal_seqs());
    GGML_ASSERT(ubatch.n_tokens == n_seq_tokens * n_seqs);

    // Helper: pin tensor to primary device when TP is active.
    // This prevents scheduler pass 5 from absorbing linear attention ops into RPC splits.
    auto pin_primary = [&](ggml_tensor * t) {
        if (tp && t) { ggml_backend_sched_set_tensor_backend(sched, t, get_tp_backend(0)); }
    };

    // Input projections
    auto qkvz = build_qkvz(cur, il);
    ggml_tensor * qkv_mixed = qkvz.first;
    ggml_tensor * z         = qkvz.second;
    pin_primary(qkv_mixed);
    pin_primary(z);

    ggml_tensor * beta = build_lora_mm(model.layers[il].ssm_beta, cur);
    beta = ggml_reshape_4d(ctx0, beta, 1, num_v_heads, n_seq_tokens, n_seqs);
    cb(beta, "beta", il);
    pin_primary(beta);

    beta = ggml_sigmoid(ctx0, beta);
    pin_primary(beta);

    ggml_tensor * alpha = build_lora_mm(model.layers[il].ssm_alpha, cur);
    alpha = ggml_cont_3d(ctx0, alpha, num_v_heads, n_seq_tokens, n_seqs);
    cb(alpha, "alpha", il);
    pin_primary(alpha);

    ggml_tensor * alpha_biased   = ggml_add(ctx0, alpha, model.layers[il].ssm_dt);
    pin_primary(alpha_biased);
    ggml_tensor * alpha_softplus = ggml_softplus(ctx0, alpha_biased);
    cb(alpha_softplus, "a_softplus", il);
    pin_primary(alpha_softplus);

    ggml_tensor * gate = ggml_mul(ctx0, alpha_softplus, model.layers[il].ssm_a);  // -A_log.exp() * softplus
    cb(gate, "gate", il);
    pin_primary(gate);

    gate = ggml_reshape_4d(ctx0, gate, 1, num_v_heads, n_seq_tokens, n_seqs);

    // Get convolution states from cache
    ggml_tensor * conv_states_all = mctx_cur->get_r_l(il);
    ggml_tensor * ssm_states_all  = mctx_cur->get_s_l(il);

    // Build the convolution states tensor
    ggml_tensor * conv_states = build_rs(inp, conv_states_all, hparams.n_embd_r(), n_seqs);
    cb(conv_states, "conv_states", il);
    // When TP is active, pin state cache ops to primary device to prevent
    // scheduler expansion from leaking RPC assignment to state tensors
    if (tp) {
        ggml_backend_sched_set_tensor_backend(sched, conv_states, get_tp_backend(0));
    }

    // Calculate convolution kernel size
    ggml_tensor * conv_kernel      = model.layers[il].ssm_conv1d;
    const int64_t conv_kernel_size = conv_kernel->ne[0];
    const int64_t conv_channels    = d_inner + 2 * hparams.ssm_n_group * hparams.ssm_d_state;

    conv_states = ggml_reshape_3d(ctx0, conv_states, conv_kernel_size - 1, conv_channels, n_seqs);
    cb(conv_states, "conv_states_reshaped", il);

    qkv_mixed = ggml_transpose(ctx0, qkv_mixed);
    cb(qkv_mixed, "qkv_mixed_transposed", il);

    ggml_tensor * conv_input = ggml_concat(ctx0, conv_states, qkv_mixed, 0);
    cb(conv_input, "conv_input", il);
    if (tp) { ggml_backend_sched_set_tensor_backend(sched, conv_input, get_tp_backend(0)); }

    // Update convolution state cache
    // Extract the last (conv_kernel_size - 1) states from conv_input
    ggml_tensor * last_conv_states =
        ggml_view_3d(ctx0, conv_input, conv_kernel_size - 1, conv_channels, n_seqs, conv_input->nb[1],
                     conv_input->nb[2], (conv_input->ne[0] - conv_states->ne[0]) * ggml_element_size(conv_input));
    cb(last_conv_states, "last_conv_states", il);

    ggml_tensor * state_update_target =
        ggml_view_1d(ctx0, conv_states_all, (conv_kernel_size - 1) * conv_channels * n_seqs,
                     kv_head * (conv_kernel_size - 1) * conv_channels * ggml_element_size(conv_states_all));
    cb(state_update_target, "state_update_target", il);

    ggml_tensor * conv_cpy = ggml_cpy(ctx0, last_conv_states, state_update_target);
    if (tp) { ggml_backend_sched_set_tensor_backend(sched, conv_cpy, get_tp_backend(0)); }
    ggml_build_forward_expand(gf, conv_cpy);
    cb(conv_states_all, "conv_states_updated", il);

    ggml_tensor * state = build_rs(inp, ssm_states_all, hparams.n_embd_s(), n_seqs);
    if (tp) { ggml_backend_sched_set_tensor_backend(sched, state, get_tp_backend(0)); }
    state = ggml_reshape_4d(ctx0, state, head_v_dim, head_v_dim, num_v_heads, n_seqs);
    cb(state, "state_predelta", il);

    ggml_tensor * conv_output_proper = ggml_ssm_conv(ctx0, conv_input, conv_kernel);
    cb(conv_output_proper, "conv_output_raw", il);
    if (tp) { ggml_backend_sched_set_tensor_backend(sched, conv_output_proper, get_tp_backend(0)); }

    ggml_tensor * conv_output_silu = ggml_silu(ctx0, conv_output_proper);
    cb(conv_output_silu, "conv_output_silu", il);
    pin_primary(conv_output_silu);

    ggml_tensor * conv_qkv_mix = conv_output_silu;

    // Calculate the total conv dimension
    int64_t qkv_dim = head_k_dim * num_k_heads * 2 + head_v_dim * num_v_heads;
    int64_t nb1_qkv = ggml_row_size(conv_qkv_mix->type, qkv_dim);

    // Extract the convolved Q, K, V from conv_output
    ggml_tensor * q_conv = ggml_view_4d(ctx0, conv_qkv_mix, head_k_dim, num_k_heads, n_seq_tokens, n_seqs,
            ggml_row_size(conv_qkv_mix->type, head_k_dim),
            nb1_qkv,
            nb1_qkv * n_seq_tokens,
            0);

    ggml_tensor * k_conv = ggml_view_4d(ctx0, conv_qkv_mix, head_k_dim, num_k_heads, n_seq_tokens, n_seqs,
            ggml_row_size(conv_qkv_mix->type, head_k_dim),
            nb1_qkv,
            nb1_qkv * n_seq_tokens,
            head_k_dim * num_k_heads * ggml_element_size(conv_qkv_mix));

    ggml_tensor * v_conv = ggml_view_4d(ctx0, conv_qkv_mix, head_v_dim, num_v_heads, n_seq_tokens, n_seqs,
            ggml_row_size(conv_qkv_mix->type, head_v_dim),
            nb1_qkv,
            nb1_qkv * n_seq_tokens,
            ggml_row_size(conv_qkv_mix->type, 2 * head_k_dim * num_k_heads));

    cb(q_conv, "q_conv", il);
    cb(k_conv, "k_conv", il);
    cb(v_conv, "v_conv", il);

    const float eps_norm = hparams.f_norm_rms_eps;

    q_conv = ggml_l2_norm(ctx0, q_conv, eps_norm);
    pin_primary(q_conv);
    k_conv = ggml_l2_norm(ctx0, k_conv, eps_norm);
    pin_primary(k_conv);

    //q_conv = ggml_cont_4d(ctx0, q_conv, head_k_dim, num_k_heads, n_seq_tokens, n_seqs);
    //k_conv = ggml_cont_4d(ctx0, k_conv, head_k_dim, num_k_heads, n_seq_tokens, n_seqs);
    //v_conv = ggml_cont_4d(ctx0, v_conv, head_v_dim, num_v_heads, n_seq_tokens, n_seqs);

    // if head keys and value keys are different, repeat to force tensors into matching shapes
    if (num_k_heads != num_v_heads) {
        GGML_ASSERT(num_v_heads % num_k_heads == 0);
        // TODO: try to avoid these explicit repeats by utilizing op broadcast
        q_conv = ggml_repeat_4d(ctx0, q_conv, head_k_dim, num_v_heads, n_seq_tokens, n_seqs);
        pin_primary(q_conv);
        k_conv = ggml_repeat_4d(ctx0, k_conv, head_k_dim, num_v_heads, n_seq_tokens, n_seqs);
        pin_primary(k_conv);
    }

    cb(q_conv, "q_conv_predelta", il);
    cb(k_conv, "k_conv_predelta", il);
    cb(v_conv, "v_conv_predelta", il);

    // Choose between build_delta_net_chunking, build_delta_net_recurrent, and build_delta_net_autoregressive based on n_tokens
    std::pair<ggml_tensor *, ggml_tensor *> attn_out; // pair of (output, new_state)
    if (n_seq_tokens == 1) {
        attn_out = build_delta_net_autoregressive(q_conv, k_conv, v_conv, gate, beta, state, il);
    } else {
        attn_out = build_delta_net_chunking(q_conv, k_conv, v_conv, gate, beta, state, il);
    }
    ggml_tensor * output    = attn_out.first;
    ggml_tensor * new_state = attn_out.second;
    cb(output, "attn_output", il);
    cb(new_state, "new_state", il);
    pin_primary(output);
    pin_primary(new_state);

    // Update the recurrent states
    {
        ggml_tensor * ssm_cpy = ggml_cpy(ctx0, new_state,
                ggml_view_1d(ctx0, ssm_states_all, hparams.n_embd_s() * n_seqs,
                    kv_head * hparams.n_embd_s() * ggml_element_size(ssm_states_all)));
        if (tp) { ggml_backend_sched_set_tensor_backend(sched, ssm_cpy, get_tp_backend(0)); }
        ggml_build_forward_expand(gf, ssm_cpy);
    }

    // z: [head_dim, n_heads, n_tokens, n_seqs] -> [n_heads * n_tokens * n_seqs, head_dim]
    ggml_tensor * z_2d = ggml_reshape_4d(ctx0, z, head_v_dim, num_v_heads, n_seq_tokens, n_seqs);

    // Apply gated normalization: self.norm(core_attn_out, z)
    ggml_tensor * attn_out_norm = build_norm_gated(output, model.layers[il].ssm_norm, z_2d, il);
    pin_primary(attn_out_norm);

    // Final reshape: [head_dim, n_heads, n_tokens, n_seqs] -> [n_tokens, n_seqs, n_heads * head_dim]
    ggml_tensor * final_output = ggml_reshape_3d(ctx0, attn_out_norm, head_v_dim * num_v_heads, n_seq_tokens, n_seqs);
    cb(final_output, "final_output", il);

    // Output projection
    cur = build_lora_mm(model.layers[il].ssm_out, final_output);
    cb(cur, "linear_attn_out", il);
    if (tp) { ggml_backend_sched_set_tensor_backend(sched, cur, get_tp_backend(0)); }

    // Reshape back to original dimensions
    cur = ggml_reshape_2d(ctx0, cur, n_embd, n_seq_tokens * n_seqs);

    return cur;
}

ggml_tensor * llm_build_qwen35moe ::build_layer_ffn(ggml_tensor * cur, const int il) {
    // Check if this is an MoE layer
    GGML_ASSERT(model.layers[il].ffn_gate_inp != nullptr);

    // MoE routed experts
    ggml_tensor * moe_out;
    if (tp && model.layers[il].ffn_gate_exps_tp[0]) {
        int64_t split[2] = { model.tp_expert_split[0], model.tp_expert_split[1] };
        moe_out = build_moe_ffn_tp(cur,
            model.layers[il].ffn_gate_inp,
            model.layers[il].ffn_up_exps_tp,
            model.layers[il].ffn_gate_exps_tp,
            model.layers[il].ffn_down_exps_tp,
            nullptr,
            n_expert, n_expert_used, split,
            LLM_FFN_SILU, true,
            false, 0.0, LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX, il);
    } else {
        moe_out = build_moe_ffn(cur,
            model.layers[il].ffn_gate_inp, model.layers[il].ffn_up_exps,
            model.layers[il].ffn_gate_exps, model.layers[il].ffn_down_exps,
            nullptr,
            n_expert, n_expert_used, LLM_FFN_SILU,
            true, false, 0.0, LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX, il);
    }
    cb(moe_out, "ffn_moe_out", il);

    // Add shared experts if present - following Qwen3Next reference implementation
    if (model.layers[il].ffn_up_shexp != nullptr) {
        ggml_tensor * ffn_shexp;

        // shared experts always run on primary device (no TP split for v1)
        ffn_shexp = build_ffn(cur,
            model.layers[il].ffn_up_shexp, NULL, NULL,
            model.layers[il].ffn_gate_shexp, NULL, NULL,
            model.layers[il].ffn_down_shexp, NULL, NULL,
            NULL,
            LLM_FFN_SILU, LLM_FFN_PAR, il);
        cb(ffn_shexp, "ffn_shexp", il);

        // Apply shared expert gating as in the reference implementation
        // The shared expert has its own gate that is sigmoided
        // Note: ffn_gate_inp_shexp is the shared expert gate (outputs 1 value per token)
        ggml_tensor * shared_gate = build_lora_mm(model.layers[il].ffn_gate_inp_shexp, cur);
        cb(shared_gate, "shared_expert_gate", il);

        // Apply sigmoid to the gate
        shared_gate = ggml_sigmoid(ctx0, shared_gate);
        cb(shared_gate, "shared_expert_gate_sigmoid", il);

        // Apply the gate to the shared expert output
        ffn_shexp = ggml_mul(ctx0, ffn_shexp, shared_gate);
        cb(ffn_shexp, "ffn_shexp_gated", il);

        cur = ggml_add(ctx0, moe_out, ffn_shexp);
        cb(cur, "ffn_out", il);
    } else {
        cur = moe_out;
    }

    return cur;
}
