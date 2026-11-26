export function buildStreamlitTabsPart2(): string {
  return `
# ==============================================================================
# TAB 1: TRAINING (EXACT 5-STEP STUDIO PARITY)
# ==============================================================================
if active_tab == "Training":
    _page_header(
        "Training",
        "Set up your voice dataset, extract acoustic features, and run RVC training.",
    )

    cfg_now = _load_config()
    prec_val = str(cfg_now.get("precision", st.session_state.get("nr_precision", "fp32")))
    gpu_ids_val = str(st.session_state.get("nr_gpu_num", "0-1"))
    gpu_cnt_cur = max(1, len([g for g in gpu_ids_val.split("-") if g.strip()]))
    bs_cur = int(st.session_state.get("nr_batch_size", 2))
    sr_cur = int(st.session_state.get("nr_sr", 32000))
    voc_cur = str(st.session_state.get("nr_vocoder", "HiFi-GAN"))
    tot_cur = int(st.session_state.get("nr_total_epoch", 200))
    sev_cur = int(st.session_state.get("nr_save_every", 5))
    ckpt_cur = bool(st.session_state.get("nr_checkpointing", False))

    sr_factor = 1.28 if sr_cur == 48000 else (1.12 if sr_cur == 40000 else 1.0)
    voc_factor = 1.45 if voc_cur == "RefineGAN" else (1.22 if voc_cur == "MRF HiFi-GAN" else 1.0)
    fp_factor = 1.0 if prec_val == "fp16" else 1.55
    ck_factor = 0.78 if ckpt_cur else 1.0
    est_vram = round((2.6 * sr_factor * voc_factor) + bs_cur * (0.72 * sr_factor * voc_factor * fp_factor * ck_factor), 2)
    vram_status = "Risk of OOM" if est_vram > 14.5 else ("Tight" if est_vram > 12.6 else "Optimal")
    save_events = max(1, tot_cur // max(1, sev_cur))
    est_disk = round(3.2 + (0.58 if st.session_state.get("nr_save_only_latest", True) else save_events * 0.58) + (save_events * 0.055 if st.session_state.get("nr_save_every_weights", True) else 0.055), 2)
    est_hrs = round((tot_cur * 46.0 * (0.62 if gpu_cnt_cur >= 2 else 1.0) * (1.35 if voc_cur == "RefineGAN" else (1.15 if voc_cur == "MRF HiFi-GAN" else 1.0)) * (0.78 if prec_val == "fp16" else 1.0)) / 3600.0, 1)

    with st.expander(f"Active Model & VRAM Info · {est_vram} GB / 16.0 GB T4 ({vram_status}) · ~{est_hrs}h", expanded=False):
        vi1, vi2, vi3, vi4 = st.columns(4, gap="small")
        vi1.markdown(f'<div class="nr-metric-box"><div class="nr-metric-k">Precision</div><div class="nr-metric-v">{html.escape(prec_val.upper())}</div></div>', unsafe_allow_html=True)
        vi2.markdown(f'<div class="nr-metric-box"><div class="nr-metric-k">Effective Batch</div><div class="nr-metric-v">{bs_cur * gpu_cnt_cur} ({bs_cur} × {gpu_cnt_cur} GPU)</div></div>', unsafe_allow_html=True)
        vi3.markdown(f'<div class="nr-metric-box"><div class="nr-metric-k">Estimated Disk</div><div class="nr-metric-v">{est_disk} / 20.0 GB</div></div>', unsafe_allow_html=True)
        vi4.markdown(f'<div class="nr-metric-box"><div class="nr-metric-k">Detected Hardware</div><div class="nr-metric-v">{html.escape(_gpu_info_lines(gpu_ids_val).replace(chr(10), " · "))}</div></div>', unsafe_allow_html=True)

    # --- STEP 1: Model Settings ---
    with _nr_card("nr_train_step_1"):
        _step_head(1, "Model Settings", "Give your voice project a name, select target sample fidelity, and assign your synthesis vocoder.")
        s1_c1, s1_c2 = st.columns(2, gap="medium")
        with s1_c1:
            st.markdown('<div class="nr-field-title">Model Name</div><div class="nr-field-help">Identifier used for saved checkpoints, index files, and log directories.</div>', unsafe_allow_html=True)
            if "nr_model_name_inp" not in st.session_state:
                st.session_state["nr_model_name_inp"] = st.session_state["nr_model_name"]
            st.text_input(
                "Model Name",
                placeholder="Enter model name (e.g. VocalModel-32k)",
                label_visibility="collapsed",
                key="nr_model_name_inp",
            )
            st.session_state["nr_model_name"] = st.session_state["nr_model_name_inp"]
        with s1_c2:
            st.markdown('<div class="nr-field-title">Sampling Rate</div><div class="nr-field-help">Target audio resolution in Hz for preprocessed slices and model synthesis.</div>', unsafe_allow_html=True)
            sr_opts = [32000, 40000, 48000]
            cur_sr_idx = sr_opts.index(int(st.session_state["nr_sr"])) if int(st.session_state["nr_sr"]) in sr_opts else 0
            st.session_state["nr_sr"] = st.selectbox(
                "Sampling Rate",
                options=sr_opts,
                index=cur_sr_idx,
                label_visibility="collapsed",
                key="nr_sr_sel",
            )

        st.markdown(
            '<div class="nr-field-title">Vocoder</div>'
            '<div class="nr-field-help">Waveform synthesizer architecture: HiFi-GAN works across all RVC players; MRF HiFi-GAN boosts harmonic detail for Applio clients; RefineGAN maximizes clarity with extra VRAM demand.</div>',
            unsafe_allow_html=True,
        )
        voc_opts = ["HiFi-GAN", "MRF HiFi-GAN", "RefineGAN"]
        cur_voc_idx = voc_opts.index(st.session_state["nr_vocoder"]) if st.session_state["nr_vocoder"] in voc_opts else 0
        st.session_state["nr_vocoder"] = st.selectbox(
            "Vocoder",
            options=voc_opts,
            index=cur_voc_idx,
            label_visibility="collapsed",
            key="nr_vocoder_sel",
        )

        with st.expander("Advanced Settings", expanded=False):
            cpu_max = max(4, os.cpu_count() or 4)
            st.session_state["nr_cpu_cores"] = _synced_slider_num(
                "CPU Cores",
                "Parallel CPU worker threads allocated for audio slicing and pitch extraction. Using all 4 Kaggle vCPUs keeps preprocessing fast.",
                "nr_cpu_cores",
                1,
                cpu_max,
                min(4, cpu_max),
            )
            a1_l, a1_r = st.columns(2, gap="medium")
            with a1_l:
                st.markdown('<div class="nr-field-title">GPU Number</div><div class="nr-field-help">Target CUDA device IDs separated by hyphens (for dual T4 sessions keep 0-1).</div>', unsafe_allow_html=True)
                if "nr_gpu_num_inp" not in st.session_state:
                    st.session_state["nr_gpu_num_inp"] = st.session_state["nr_gpu_num"]
                st.text_input(
                    "GPU Number",
                    label_visibility="collapsed",
                    key="nr_gpu_num_inp",
                )
                st.session_state["nr_gpu_num"] = st.session_state["nr_gpu_num_inp"]
            with a1_r:
                st.markdown('<div class="nr-field-title">GPU Information</div><div class="nr-field-help">Detected CUDA accelerator allocation for this session.</div>', unsafe_allow_html=True)
                _output_box("Detected CUDA Devices", _gpu_info_lines(st.session_state["nr_gpu_num"]))

            existing_logs = _list_models()
            if existing_logs:
                st.markdown('<div class="nr-field-title">Resume Existing Checkpoint Folder (Optional)</div>', unsafe_allow_html=True)

                def _on_pick_resume_model() -> None:
                    picked_m = st.session_state.get("nr_resume_existing_sel", "")
                    if picked_m:
                        st.session_state["nr_model_name"] = picked_m
                        st.session_state["nr_model_name_inp"] = picked_m
                        st.session_state["nr_resume_existing_sel"] = ""

                st.selectbox(
                    "Resume Existing Model",
                    options=[""] + existing_logs,
                    index=0,
                    format_func=lambda x: "Keep current model name..." if x == "" else x,
                    label_visibility="collapsed",
                    on_change=_on_pick_resume_model,
                    key="nr_resume_existing_sel",
                )

    # --- STEP 2: Prepare Dataset ---
    with _nr_card("nr_train_step_2"):
        _step_head(2, "Prepare Dataset", "Link your Kaggle dataset folder and split clean vocal segments for training.")
        detected_datasets = _list_datasets()
        if "nr_ds_path_inp" not in st.session_state:
            st.session_state["nr_ds_path_inp"] = st.session_state["nr_dataset_path"]
        cur_ds_val = str(st.session_state.get("nr_ds_path_inp", "")).strip()
        if (not cur_ds_val or "your-username" in cur_ds_val or not Path(cur_ds_val).is_dir()) and detected_datasets:
            st.session_state["nr_dataset_path"] = detected_datasets[0]
            st.session_state["nr_ds_path_inp"] = detected_datasets[0]

        st.markdown('<div class="nr-field-title">Dataset Path</div><div class="nr-field-help">Mounted Kaggle input path containing your dry vocal recordings.</div>', unsafe_allow_html=True)
        st.text_input(
            "Dataset Path",
            placeholder="/kaggle/input/datasets/your-username/voice-dataset",
            label_visibility="collapsed",
            key="nr_ds_path_inp",
        )
        st.session_state["nr_dataset_path"] = st.session_state["nr_ds_path_inp"]
        if detected_datasets:
            st.caption(f"✔ Auto-detected Kaggle dataset folder ({len(detected_datasets)} found): {st.session_state['nr_dataset_path']}")

        ds_l, ds_r = st.columns([3, 1.3], gap="small")
        with ds_l:
            st.session_state["nr_dataset_creator"] = st.checkbox(
                "Dataset Creator",
                value=bool(st.session_state["nr_dataset_creator"]),
                key="nr_ds_creator_chk",
            )
        with ds_r:
            if st.button("Refresh models and datasets", use_container_width=True, key="nr_refresh_ds_btn"):
                _clear_fs_caches()
                refreshed_ds = _list_datasets()
                cur_ds_chk = str(st.session_state.get("nr_dataset_path", "")).strip()
                if refreshed_ds and ("your-username" in cur_ds_chk or not cur_ds_chk or not Path(cur_ds_chk).is_dir()):
                    st.session_state["nr_dataset_path"] = refreshed_ds[0]
                    st.session_state["nr_ds_path_inp"] = refreshed_ds[0]
                st.session_state["nr_prep_log"] = f"Refreshed dataset & model scan ({len(refreshed_ds)} dataset folder(s) found)."
                st.rerun()

        if st.session_state["nr_dataset_creator"]:
            with st.container(border=True):
                dc_name = st.text_input("Dataset Name", value=re.sub(r"[^a-zA-Z0-9_-]+", "_", st.session_state["nr_model_name"] or "my_dataset"), key="nr_dc_name")
                dc_files = st.file_uploader(
                    "Upload audio clips for your dataset",
                    type=["wav", "mp3", "flac", "ogg", "m4a"],
                    accept_multiple_files=True,
                    key="nr_dc_uploader",
                )
                if dc_files and st.button("Upload Audio Files", type="secondary", key="nr_dc_save_btn"):
                    target_ds = Path(now_dir) / "assets" / "datasets" / (re.sub(r"[^a-zA-Z0-9_-]+", "_", dc_name) or "my_dataset")
                    target_ds.mkdir(parents=True, exist_ok=True)
                    for uf in dc_files:
                        (target_ds / uf.name).write_bytes(uf.getvalue())
                    _clear_fs_caches()
                    st.session_state["nr_dataset_path"] = str(target_ds)
                    st.session_state["nr_ds_path_inp"] = str(target_ds)
                    st.session_state["nr_prep_log"] = f"Saved {len(dc_files)} clip(s) into {target_ds}."
                    st.rerun()

        with st.expander("Advanced Settings", expanded=False):
            if len(detected_datasets) > 1:
                st.markdown('<div class="nr-field-title">Detected Mounted Kaggle Datasets</div><div class="nr-field-help">Switch between mounted Kaggle datasets automatically without manual copying.</div>', unsafe_allow_html=True)

                def _on_pick_detected_ds() -> None:
                    picked_d = st.session_state.get("nr_pick_detected_ds", "")
                    if picked_d:
                        st.session_state["nr_dataset_path"] = picked_d
                        st.session_state["nr_ds_path_inp"] = picked_d
                        st.session_state["nr_pick_detected_ds"] = ""

                st.selectbox(
                    "Detected Kaggle Dataset",
                    options=[""] + detected_datasets,
                    index=0,
                    format_func=lambda x: "Keep current dataset path..." if x == "" else x,
                    label_visibility="collapsed",
                    on_change=_on_pick_detected_ds,
                    key="nr_pick_detected_ds",
                )

            st.markdown('<div class="nr-field-title">Audio cutting</div><div class="nr-field-help">Choose how long recordings are divided: Automatic detects quiet gaps dynamically, Simple cuts fixed chunks, and Skip leaves pre-split clips untouched.</div>', unsafe_allow_html=True)
            cut_opts = ["Automatic", "Simple", "Skip"]
            cur_cut_idx = cut_opts.index(st.session_state["nr_audio_cutting"]) if st.session_state["nr_audio_cutting"] in cut_opts else 0
            st.session_state["nr_audio_cutting"] = st.selectbox("Audio cutting", options=cut_opts, index=cur_cut_idx, label_visibility="collapsed", key="nr_cut_sel")

            c2_l, c2_r = st.columns(2, gap="medium")
            with c2_l:
                st.session_state["nr_chunk_len"] = st.slider(
                    "Chunk length (sec)",
                    min_value=0.5,
                    max_value=5.0,
                    value=float(st.session_state["nr_chunk_len"]),
                    step=0.1,
                    help="Duration of each split audio window when Simple cutting is selected.",
                    key="nr_chunk_sl",
                )
            with c2_r:
                st.session_state["nr_overlap_len"] = st.slider(
                    "Overlap length (sec)",
                    min_value=0.0,
                    max_value=0.4,
                    value=float(st.session_state["nr_overlap_len"]),
                    step=0.05,
                    help="Crossfade overlap between consecutive Simple slices to avoid abrupt cuts.",
                    key="nr_overlap_sl",
                )

            ts_l, ts_r = st.columns([2, 1.5], gap="medium")
            with ts_l:
                st.session_state["nr_trunc_sil"] = st.checkbox(
                    "Truncate silence",
                    value=bool(st.session_state["nr_trunc_sil"]),
                    key="nr_trunc_chk",
                )
            with ts_r:
                st.session_state["nr_sil_ms"] = st.number_input(
                    "Silence duration (ms)",
                    min_value=50,
                    max_value=3000,
                    step=50,
                    value=int(st.session_state["nr_sil_ms"]),
                    disabled=not bool(st.session_state["nr_trunc_sil"]),
                    key="nr_sil_ms_inp",
                )

            ef_l, ef_r = st.columns(2, gap="medium")
            with ef_l:
                st.session_state["nr_noise_filter"] = st.checkbox(
                    "Noise filter — Applies spectral denoising to raw clips (keep off if clean)",
                    value=bool(st.session_state["nr_noise_filter"]),
                    key="nr_nf_chk",
                )
            with ef_r:
                st.session_state["nr_loud_norm"] = st.checkbox(
                    "Loudness normalization (-14 LUFS) — Levels every clip consistently without clipping",
                    value=bool(st.session_state["nr_loud_norm"]),
                    key="nr_ln_chk",
                )

        if st.button(_tr("Preprocess Dataset"), type="primary", use_container_width=True, key="nr_btn_preprocess"):
            m_clean = st.session_state["nr_model_name"].strip()
            ds_clean = st.session_state["nr_dataset_path"].strip()
            if not m_clean:
                st.warning("Enter a Model Name in Step 1 first.")
            elif not ds_clean:
                st.warning("Enter a Dataset Path first.")
            else:
                use_fx = bool(st.session_state["nr_noise_filter"] or st.session_state["nr_loud_norm"])
                out_txt = _start_core_task_bg(
                    "preprocess",
                    "run_preprocess_script",
                    m_clean,
                    {
                        "model_name": m_clean,
                        "dataset_path": ds_clean,
                        "sample_rate": int(st.session_state["nr_sr"]),
                        "cpu_cores": int(st.session_state["nr_cpu_cores"]),
                        "cut_preprocess": st.session_state["nr_audio_cutting"],
                        "process_effects": use_fx,
                        "noise_reduction": bool(st.session_state["nr_noise_filter"]),
                        "noise_reduction_strength": 0.7,
                        "chunk_len": float(st.session_state["nr_chunk_len"]),
                        "overlap_len": float(st.session_state["nr_overlap_len"]),
                        "normalization_mode": "post" if st.session_state["nr_loud_norm"] else "none",
                        "truncate_silence": bool(st.session_state["nr_trunc_sil"]),
                        "silence_duration_ms": int(st.session_state["nr_sil_ms"]),
                    },
                    "nr_prep_log",
                    "Preprocess Complete",
                    f"Dataset sliced & ready for {m_clean} ({st.session_state['nr_sr']} Hz).",
                )
                st.session_state["nr_prep_log"] = out_txt
                st.rerun()

        with _nr_box("nr_prep_out_box"):
            def _render_prep_live_body() -> None:
                _, done_p = _poll_core_tasks()
                _output_box("Preprocess output", st.session_state["nr_prep_log"])
                if done_p and _HAS_FRAGMENT:
                    try:
                        st.rerun(scope="app")
                    except TypeError:
                        st.rerun()

            if _HAS_FRAGMENT:
                @st.fragment(run_every=1.5 if _is_task_running("preprocess") else None)
                def _prep_live_frag() -> None:
                    _render_prep_live_body()

                _prep_live_frag()
            else:
                _render_prep_live_body()

    # --- STEP 3: Extract Features ---
    with _nr_card("nr_train_step_3"):
        _step_head(3, "Extract Features", "Compute pitch tracks and speaker embeddings from your sliced vocals.")
        e1_l, e1_r = st.columns(2, gap="medium")
        with e1_l:
            st.markdown('<div class="nr-field-title">Pitch extraction algorithm</div><div class="nr-field-help">F0 tracker used to capture vocal melody. RMVPE offers the sharpest accuracy across speech and singing.</div>', unsafe_allow_html=True)
            f0_opts = ["rmvpe", "fcpe", "swift-f0", "crepe"]
            st.session_state["nr_f0"] = st.selectbox("Pitch extraction algorithm", options=f0_opts, index=f0_opts.index(st.session_state["nr_f0"]) if st.session_state["nr_f0"] in f0_opts else 0, label_visibility="collapsed", key="nr_f0_sel")
        with e1_r:
            st.markdown('<div class="nr-field-title">Embedder Model</div><div class="nr-field-help">Speech encoder that extracts phonetic timbre representations from audio.</div>', unsafe_allow_html=True)
            emb_opts = ["contentvec", "spin", "chinese-hubert-base"]
            st.session_state["nr_embedder"] = st.selectbox("Embedder Model", options=emb_opts, index=emb_opts.index(st.session_state["nr_embedder"]) if st.session_state["nr_embedder"] in emb_opts else 0, label_visibility="collapsed", key="nr_emb_sel")

        st.session_state["nr_silent_files"] = _synced_slider_num(
            "Silent training files",
            "Injects blank silent samples to help prevent breath artifacts during pauses. Leave at 0 if your dataset already includes natural quiet gaps.",
            "nr_silent_files",
            0,
            10,
            0,
        )

        if st.button(_tr("Extract Features"), type="primary", use_container_width=True, key="nr_btn_extract"):
            m_clean = st.session_state["nr_model_name"].strip()
            if not m_clean:
                st.warning("Enter a Model Name in Step 1 first.")
            else:
                out_txt = _start_core_task_bg(
                    "extract",
                    "run_extract_script",
                    m_clean,
                    {
                        "model_name": m_clean,
                        "f0_method": st.session_state["nr_f0"],
                        "cpu_cores": int(st.session_state["nr_cpu_cores"]),
                        "gpu": str(st.session_state["nr_gpu_num"]).strip() or "0",
                        "sample_rate": int(st.session_state["nr_sr"]),
                        "embedder_model": st.session_state["nr_embedder"],
                        "embedder_model_custom": None,
                        "include_mutes": int(st.session_state["nr_silent_files"]),
                    },
                    "nr_ext_log",
                    "Feature Extraction Complete",
                    f"Pitch ({st.session_state['nr_f0']}) & {st.session_state['nr_embedder']} embeddings ready for {m_clean}.",
                )
                st.session_state["nr_ext_log"] = out_txt
                st.rerun()

        with _nr_box("nr_ext_out_box"):
            def _render_ext_live_body() -> None:
                _, done_e = _poll_core_tasks()
                _output_box("Feature extraction output", st.session_state["nr_ext_log"])
                if done_e and _HAS_FRAGMENT:
                    try:
                        st.rerun(scope="app")
                    except TypeError:
                        st.rerun()

            if _HAS_FRAGMENT:
                @st.fragment(run_every=1.5 if _is_task_running("extract") else None)
                def _ext_live_frag() -> None:
                    _render_ext_live_body()

                _ext_live_frag()
            else:
                _render_ext_live_body()

    # --- STEP 4: Train Model ---
    with _nr_card("nr_train_step_4"):
        _step_head(4, "Train Model", "Dial in batch size, checkpoint frequency, and total training epochs before launching.")

        st.session_state["nr_batch_size"] = _synced_slider_num(
            "Batch Size",
            "Target batch size. 1-GPU Applio Batch Parity (ON in Settings -> Training) splits >= 4 across dual T4s so under-300 epoch runs match Original Applio step count.",
            "nr_batch_size",
            1,
            64,
            2,
        )
        bs_now = int(st.session_state["nr_batch_size"])
        parity_now = bool(st.session_state.get("nr_batch_parity", True))
        is_parity_split = parity_now and gpu_cnt_cur >= 2 and bs_now >= 4
        per_gpu_bs = max(2, bs_now // gpu_cnt_cur) if is_parity_split else bs_now
        eff_now = per_gpu_bs * gpu_cnt_cur
        eff_note = f"Split {per_gpu_bs}/GPU × {gpu_cnt_cur} T4s = {eff_now} Total (1-GPU Parity)" if is_parity_split else (f"Dual T4 Effective: {eff_now}" if gpu_cnt_cur >= 2 else f"Effective: {eff_now}")
        st.markdown(
            f'<div style="display:flex;justify-content:space-between;align-items:center;background:#100506;border:1px solid #47161D;border-radius:8px;padding:7px 12px;margin:2px 0 10px 0;font-family:JetBrains Mono,monospace;font-size:12px">'
            f'<span style="color:#DFBFC3">Active Batch Allocation ({gpu_cnt_cur} GPU{"s" if gpu_cnt_cur != 1 else ""})</span>'
            f'<span style="color:#FDF7F7;font-weight:700">Batch {bs_now} · <span style="color:#E51923">{eff_note}</span></span></div>',
            unsafe_allow_html=True,
        )
        st.session_state["nr_save_every"] = _synced_slider_num(
            "Save Every Epoch",
            "Interval of completed epochs between saved checkpoint snapshots.",
            "nr_save_every",
            1,
            100,
            5,
        )
        st.session_state["nr_total_epoch"] = _synced_slider_num(
            "Total Epoch",
            "Target number of full dataset passes before training stops automatically.",
            "nr_total_epoch",
            1,
            10000,
            200,
        )

        with st.expander("Advanced Settings", expanded=False):
            st.session_state["nr_turbo_training"] = st.checkbox(
                "Lossless Turbo Training — Always ON by default (adaptive RAM cache, IEEE-exact foreach AdamW, cuDNN benchmark, DDP bucket view)",
                value=bool(st.session_state.get("nr_turbo_training", True)),
                key="nr_t_turbo",
            )
            toggles_l, toggles_r = st.columns(2, gap="medium")
            with toggles_l:
                st.session_state["nr_save_only_latest"] = st.checkbox("Save Only Latest — Overwrites previous G/D resume states to prevent Kaggle disk overflow", value=bool(st.session_state["nr_save_only_latest"]), key="nr_t_latest")
                st.session_state["nr_save_every_weights"] = st.checkbox("Save Every Weights — Exports standalone inference .pth weight at each save interval", value=bool(st.session_state["nr_save_every_weights"]), key="nr_t_weights")
                st.session_state["nr_pretrained"] = st.checkbox("Pretrained — Initializes training from base RVC weights for cleaner timbre", value=bool(st.session_state["nr_pretrained"]), key="nr_t_pre")
                st.session_state["nr_fresh_cleanup"] = st.checkbox("Fresh Training — Clears prior checkpoints inside model folder before starting", value=bool(st.session_state["nr_fresh_cleanup"]), key="nr_t_fresh")
            with toggles_r:
                st.session_state["nr_cache_gpu"] = st.checkbox("Cache Dataset in GPU — Pins audio slices in VRAM to speed up small datasets", value=bool(st.session_state["nr_cache_gpu"]), key="nr_t_cache")
                st.session_state["nr_checkpointing"] = st.checkbox("Checkpointing — Reduces peak VRAM via gradient checkpointing", value=bool(st.session_state["nr_checkpointing"]), key="nr_t_ckpt")
                st.session_state["nr_shutdown_finish"] = st.checkbox("Shutdown after finishing — Powers down runtime automatically once training finishes", value=bool(st.session_state["nr_shutdown_finish"]), key="nr_t_shut")
                st.session_state["nr_custom_pre"] = st.checkbox("Custom Pretrains — Point training to custom G and D weights", value=bool(st.session_state["nr_custom_pre"]), key="nr_t_cpre")

            if st.session_state["nr_custom_pre"]:
                with st.container(border=True):
                    cp_head_l, cp_head_r = st.columns([3, 1.4], gap="small")
                    with cp_head_l:
                        st.markdown('<div class="nr-field-title">Custom G/D Pretrained Pair</div>', unsafe_allow_html=True)
                    with cp_head_r:
                        if st.button("Refresh pretrained list", key="nr_btn_refresh_pre", use_container_width=True):
                            g_in = str(st.session_state["nr_custom_g"]).strip()
                            if g_in and not str(st.session_state["nr_custom_d"]).strip():
                                inferred_d = _infer_matching_d_url(g_in)
                                if inferred_d:
                                    st.session_state["nr_custom_d"] = inferred_d
                                    st.session_state["nr_cp_d_inp"] = inferred_d
                            st.rerun()

                    cpre_dir = Path(now_dir) / "rvc" / "models" / "pretraineds" / "custom"
                    cpre_dir.mkdir(parents=True, exist_ok=True)
                    g_list = _list_custom_pretrains("G")
                    d_list = _list_custom_pretrains("D")
                    cp_l, cp_r = st.columns(2, gap="medium")
                    with cp_l:
                        if g_list:
                            def _on_pick_cp_g() -> None:
                                pg = st.session_state.get("nr_cp_g_pick", "")
                                if pg:
                                    st.session_state["nr_custom_g"] = pg
                                    st.session_state["nr_cp_g_inp"] = pg
                                    st.session_state["nr_cp_g_pick"] = ""

                            st.selectbox("Detected G Pretrain", options=[""] + g_list, format_func=lambda x: "Pick G file..." if x == "" else x, on_change=_on_pick_cp_g, key="nr_cp_g_pick")
                        if "nr_cp_g_inp" not in st.session_state:
                            st.session_state["nr_cp_g_inp"] = st.session_state["nr_custom_g"]
                        st.text_input("G Pretrained Path or URL", placeholder="rvc/models/pretraineds/custom/G.pth", key="nr_cp_g_inp")
                        st.session_state["nr_custom_g"] = st.session_state["nr_cp_g_inp"]
                        up_cp_g = st.file_uploader("Upload Custom G Pretrain (.pth or .zip)", type=["pth", "zip"], key="nr_cp_g_up")
                        if up_cp_g is not None:
                            g_sig = f"{up_cp_g.name}:{up_cp_g.size}"
                            if g_sig != st.session_state.get("nr_cp_last_g_sig"):
                                st.session_state["nr_cp_last_g_sig"] = g_sig
                                if up_cp_g.name.lower().endswith(".zip"):
                                    tmp_z = cpre_dir / up_cp_g.name
                                    tmp_z.write_bytes(up_cp_g.getvalue())
                                    with zipfile.ZipFile(tmp_z, "r") as zf:
                                        for mem in zf.namelist():
                                            if not mem.endswith("/") and Path(mem).name.lower().endswith(".pth"):
                                                out_p = cpre_dir / Path(mem).name
                                                out_p.write_bytes(zf.read(mem))
                                                rel_p = f"rvc/models/pretraineds/custom/{out_p.name}"
                                                if "D" in out_p.name.upper():
                                                    st.session_state["nr_custom_d"] = rel_p
                                                    st.session_state["nr_cp_d_inp"] = rel_p
                                                else:
                                                    st.session_state["nr_custom_g"] = rel_p
                                                    st.session_state["nr_cp_g_inp"] = rel_p
                                    tmp_z.unlink(missing_ok=True)
                                else:
                                    out_g = cpre_dir / Path(up_cp_g.name).name
                                    out_g.write_bytes(up_cp_g.getvalue())
                                    rel_g = f"rvc/models/pretraineds/custom/{out_g.name}"
                                    st.session_state["nr_custom_g"] = rel_g
                                    st.session_state["nr_cp_g_inp"] = rel_g
                                _clear_fs_caches()
                                print(f"✔ Pretrained uploaded: {up_cp_g.name}", flush=True)
                                _push_toast("Pretrained Uploaded", f"Loaded custom G pretrain ({up_cp_g.name}).")
                                st.rerun()
                    with cp_r:
                        if d_list:
                            def _on_pick_cp_d() -> None:
                                pd = st.session_state.get("nr_cp_d_pick", "")
                                if pd:
                                    st.session_state["nr_custom_d"] = pd
                                    st.session_state["nr_cp_d_inp"] = pd
                                    st.session_state["nr_cp_d_pick"] = ""

                            st.selectbox("Detected D Pretrain", options=[""] + d_list, format_func=lambda x: "Pick D file..." if x == "" else x, on_change=_on_pick_cp_d, key="nr_cp_d_pick")
                        if "nr_cp_d_inp" not in st.session_state:
                            st.session_state["nr_cp_d_inp"] = st.session_state["nr_custom_d"]
                        st.text_input("D Pretrained Path or URL (optional auto-pair)", placeholder="rvc/models/pretraineds/custom/D.pth", key="nr_cp_d_inp")
                        st.session_state["nr_custom_d"] = st.session_state["nr_cp_d_inp"]
                        up_cp_d = st.file_uploader("Upload Custom D Pretrain (.pth)", type=["pth"], key="nr_cp_d_up")
                        if up_cp_d is not None:
                            d_sig = f"{up_cp_d.name}:{up_cp_d.size}"
                            if d_sig != st.session_state.get("nr_cp_last_d_sig"):
                                st.session_state["nr_cp_last_d_sig"] = d_sig
                                out_d = cpre_dir / Path(up_cp_d.name).name
                                out_d.write_bytes(up_cp_d.getvalue())
                                rel_d = f"rvc/models/pretraineds/custom/{out_d.name}"
                                st.session_state["nr_custom_d"] = rel_d
                                st.session_state["nr_cp_d_inp"] = rel_d
                                _clear_fs_caches()
                                print(f"✔ Pretrained uploaded: {up_cp_d.name}", flush=True)
                                _push_toast("Pretrained Uploaded", f"Loaded custom D pretrain ({up_cp_d.name}).")
                                st.rerun()

            idx_opts = ["Auto", "Faiss", "KMeans"]
            st.session_state["nr_index_algo"] = st.selectbox("Index Algorithm", options=idx_opts, index=idx_opts.index(st.session_state["nr_index_algo"]) if st.session_state["nr_index_algo"] in idx_opts else 0, key="nr_idx_algo_sel")

        with _nr_card("nr_train_terms_box"):
            st.session_state.setdefault("nr_train_terms", True)
            terms_ok = st.checkbox(
                "I agree to the terms of use — Confirm that your training audio follows the usage guidelines for NoRegretsRVC.",
                key="nr_train_terms",
            )

        m_clean_cur = st.session_state["nr_model_name"].strip()
        is_running_now = _is_training_process_alive(m_clean_cur) or bool(st.session_state["nr_is_training"])

        b_start, b_idx = st.columns(2, gap="small")
        with b_start:
            if is_running_now:
                if st.button(_tr("Stop Training"), type="primary", use_container_width=True, key="nr_btn_stop_train"):
                    st.session_state["nr_trn_log"] = _stop_training(m_clean_cur)
                    st.rerun()
            else:
                if st.button(_tr("Start Training"), type="primary", use_container_width=True, key="nr_btn_start_train"):
                    if not terms_ok:
                        st.warning("Confirm the terms of use checkbox first.")
                    elif not m_clean_cur:
                        st.warning("Enter a Model Name in Step 1 first.")
                    else:
                        g_resolved = str(st.session_state["nr_custom_g"]).strip()
                        d_resolved = str(st.session_state["nr_custom_d"]).strip() or _infer_matching_d_url(g_resolved)
                        if st.session_state["nr_fresh_cleanup"]:
                            mdir = Path(now_dir) / "logs" / m_clean_cur
                            if mdir.is_dir():
                                for old_f in mdir.glob("G_*.pth"):
                                    old_f.unlink(missing_ok=True)
                                for old_f in mdir.glob("D_*.pth"):
                                    old_f.unlink(missing_ok=True)
                        bs_cfg = int(st.session_state["nr_batch_size"])
                        parity_cfg = bool(st.session_state.get("nr_batch_parity", True))
                        is_split_cfg = parity_cfg and gpu_cnt_cur >= 2 and bs_cfg >= 4
                        per_gpu_cfg = max(2, bs_cfg // gpu_cnt_cur) if is_split_cfg else bs_cfg
                        lr_scale_cfg = "1.0" if parity_cfg else ("1.35" if (gpu_cnt_cur >= 2 and bs_cfg >= 4) else "1.0")
                        msg = _start_training_bg(
                            m_clean_cur,
                            {
                                "model_name": m_clean_cur,
                                "save_every_epoch": int(st.session_state["nr_save_every"]),
                                "save_only_latest": bool(st.session_state["nr_save_only_latest"]),
                                "save_every_weights": bool(st.session_state["nr_save_every_weights"]),
                                "total_epoch": int(st.session_state["nr_total_epoch"]),
                                "sample_rate": int(st.session_state["nr_sr"]),
                                "batch_size": per_gpu_cfg,
                                "gpu": str(st.session_state["nr_gpu_num"]).strip() or "0",
                                "overtraining_detector": False,
                                "overtraining_threshold": 50,
                                "pretrained": bool(st.session_state["nr_pretrained"]),
                                "custom_pretrained": bool(st.session_state["nr_custom_pre"]),
                                "g_pretrained_path": os.path.join(now_dir, g_resolved) if g_resolved and not g_resolved.startswith(("http://", "https://", "/")) else g_resolved,
                                "d_pretrained_path": os.path.join(now_dir, d_resolved) if d_resolved and not d_resolved.startswith(("http://", "https://", "/")) else d_resolved,
                                "cleanup": bool(st.session_state["nr_fresh_cleanup"]),
                                "cache_data_in_gpu": bool(st.session_state["nr_cache_gpu"]),
                                "vocoder": st.session_state["nr_vocoder"],
                                "checkpointing": bool(st.session_state["nr_checkpointing"]),
                                "nr_ddp_lr_scale": lr_scale_cfg,
                            },
                        )
                        if msg.startswith("✖"):
                            _push_toast("Training Blocked", msg)
                        st.session_state["nr_trn_log"] = msg
                        st.rerun()
        with b_idx:
            if st.button(_tr("Generate Index"), type="secondary", use_container_width=True, key="nr_btn_gen_index"):
                if not m_clean_cur:
                    st.warning("Enter a Model Name in Step 1 first.")
                else:
                    out_txt = _start_core_task_bg(
                        "index",
                        "run_index_script",
                        m_clean_cur,
                        {
                            "model_name": m_clean_cur,
                            "index_algorithm": st.session_state["nr_index_algo"],
                        },
                        "nr_trn_log",
                        "Index Generation Complete",
                        f"FAISS feature index ready for {m_clean_cur}.",
                    )
                    st.session_state["nr_trn_log"] = out_txt
                    st.rerun()

        with _nr_box("nr_trn_live_box"):
            def _render_trn_live_body() -> None:
                _, done_t = _poll_core_tasks()
                m_now = st.session_state["nr_model_name"].strip()
                run_now = _is_training_process_alive(m_now) or bool(st.session_state["nr_is_training"])
                live_raw = _read_train_log(m_now)
                if live_raw:
                    filtered_trn = _filter_status_lines(
                        live_raw,
                        "⏳ Training in progress — monitor live epochs in Kaggle cell & progress meter above." if run_now else "Training log ready — open Diagnostics for epoch curve.",
                    )
                    if filtered_trn:
                        st.session_state["nr_trn_log"] = filtered_trn
                parsed_live = _parse_training_log(live_raw) if live_raw else None
                if parsed_live and parsed_live.get("live"):
                    lp = parsed_live["live"]
                    tot_ep_cur = max(1, int(st.session_state["nr_total_epoch"]))
                    cur_ep = min(tot_ep_cur, int(lp["epoch"]))
                    batch_tot = max(1, int(lp["batch_total"]))
                    batch_dn = min(batch_tot, int(lp["batch_done"]))
                    pct = int(lp["percent"])
                    it_s = float(lp["it_per_sec"])
                    ep_eta = max(0.0, (batch_tot - batch_dn) / max(0.1, it_s))
                    st.markdown(
                        f'<div class="nr-prog-wrap">'
                        f'<div style="display:flex;justify-content:space-between;font-family:JetBrains Mono,monospace;font-size:12px">'
                        f'<span style="color:#FDF7F7;font-weight:600">Epoch {cur_ep}/{tot_ep_cur} · Step {lp["step"]}</span>'
                        f'<span style="color:#E51923;font-weight:600">{"Running" if run_now else "Latest"}</span></div>'
                        f'<div class="nr-prog-track"><div class="nr-prog-fill" style="width:{pct}%"></div></div>'
                        f'<div style="display:flex;justify-content:space-between;font-family:JetBrains Mono,monospace;font-size:11px;color:#DFBFC3">'
                        f'<span>Batch {batch_dn}/{batch_tot} ({pct}%) · {it_s:.2f} it/s</span>'
                        f'<span style="color:#AD8288">Epoch ETA ~{ep_eta:.1f}s</span></div></div>',
                        unsafe_allow_html=True,
                    )
                    if run_now:
                        it_s_c = max(0.15, float(lp["it_per_sec"]))
                        spds_all = [float(p["speed_sec"]) for p in (parsed_live.get("points") or []) if p.get("speed_sec") is not None]
                        if spds_all:
                            tail_spds = spds_all[-8:]
                            ema_spd = tail_spds[0]
                            for _s_val in tail_spds[1:]:
                                ema_spd = 0.68 * ema_spd + 0.32 * _s_val
                            ep_sec = ema_spd
                            acc_pct = min(98, 46 + int(len(spds_all) * 4.5))
                        else:
                            ep_sec = max(1.0, batch_tot / it_s_c)
                            acc_pct = min(65, 28 + int((batch_dn / max(1, batch_tot)) * 32))
                        rem_cur_ep = max(0.0, min(ep_sec, (batch_tot - batch_dn) / it_s_c))
                        rem_sec = int(max(0, rem_cur_ep + max(0, tot_ep_cur - int(lp["epoch"])) * ep_sec))
                        mins, secs = divmod(rem_sec, 60)
                        hrs, mins = divmod(mins, 60)
                        rem_lbl = f"~{hrs} hr {mins} min" if hrs > 0 else f"~{mins} min {secs}s"
                        fin_epoch_ms = int((time.time() + rem_sec) * 1000)
                        if _st_components is not None:
                            _st_components.html(
                                f"""<!doctype html>
                                <html><head><style>
                                * {{ box-sizing: border-box; margin: 0; padding: 0; font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif; }}
                                body {{ background: transparent; color: #FDF7F7; overflow: hidden; }}
                                .eta-card {{ background: #160709; border: 1px solid #47161D; border-radius: 12px; padding: 12px 14px; }}
                                .eta-top {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px; }}
                                .eta-title {{ font-size: 12px; font-weight: 600; color: #FDF7F7; }}
                                .eta-acc {{ font-family: 'JetBrains Mono', ui-monospace, monospace; font-size: 11px; color: #FDF7F7; background: #200B0E; border: 1px solid #E51923; border-radius: 6px; padding: 2px 8px; font-variant-numeric: tabular-nums; }}
                                .eta-grid {{ display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 8px; }}
                                .eta-box {{ background: #100506; border: 1px solid #47161D; border-radius: 8px; padding: 8px 10px; }}
                                .eta-k {{ font-size: 10.5px; font-weight: 600; color: #AD8288; letter-spacing: 0.03em; }}
                                .eta-v {{ font-family: 'JetBrains Mono', ui-monospace, monospace; font-size: 13px; font-weight: 700; color: #FDF7F7; margin-top: 3px; font-variant-numeric: tabular-nums; }}
                                </style></head><body>
                                <div class="eta-card" data-nr-fin-ms="{fin_epoch_ms}">
                                  <div class="eta-top">
                                    <span class="eta-title">⏱ Estimated Finish · Updating live</span>
                                    <span class="eta-acc">ETA Accuracy: {acc_pct}%</span>
                                  </div>
                                  <div class="eta-grid">
                                    <div class="eta-box"><div class="eta-k">REMAINING TIME</div><div class="eta-v">Finishes in {html.escape(rem_lbl)}</div></div>
                                    <div class="eta-box"><div class="eta-k">YOUR LOCAL TIME</div><div class="eta-v" id="nr-fin-local-time">--:--</div></div>
                                    <div class="eta-box"><div class="eta-k">FINISH DAY</div><div class="eta-v" id="nr-fin-local-day">Today</div></div>
                                  </div>
                                </div>
                                <script>
                                  (function() {{
                                    var fin = new Date({fin_epoch_ms});
                                    var now = new Date();
                                    var tEl = document.getElementById("nr-fin-local-time");
                                    var dEl = document.getElementById("nr-fin-local-day");
                                    if (tEl) tEl.textContent = fin.toLocaleTimeString([], {{ hour: "numeric", minute: "2-digit" }});
                                    var finMid = new Date(fin.getFullYear(), fin.getMonth(), fin.getDate()).getTime();
                                    var nowMid = new Date(now.getFullYear(), now.getMonth(), now.getDate()).getTime();
                                    var diffDays = Math.round((finMid - nowMid) / 86400000);
                                    var dayStr = fin.toLocaleDateString([], {{ weekday: "short", month: "short", day: "numeric" }});
                                    var prefix = diffDays === 0 ? "Today · " : (diffDays === 1 ? "Tomorrow · " : "");
                                    if (dEl) dEl.textContent = prefix + dayStr;
                                  }})();
                                </script></body></html>""",
                                height=122,
                            )
                        else:
                            fin_ts = time.localtime(time.time() + rem_sec)
                            fin_time = time.strftime("%I:%M %p", fin_ts).lstrip("0")
                            fin_day = time.strftime("%a, %b %d", fin_ts)
                            with _nr_card("nr_train_eta_box"):
                                st.markdown(
                                    f'<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px">'
                                    f'<span style="font-size:12px;font-weight:600;color:#FDF7F7">⏱ Estimated Finish · Updating live</span>'
                                    f'<span class="nr-mono-badge" style="color:#FDF7F7;border-color:#E51923">ETA Accuracy: {acc_pct}%</span></div>',
                                    unsafe_allow_html=True,
                                )
                                e_c1, e_c2, e_c3 = st.columns(3, gap="small")
                                e_c1.markdown(f'<div class="nr-metric-box"><div class="nr-metric-k">REMAINING TIME</div><div class="nr-metric-v">Finishes in {rem_lbl}</div></div>', unsafe_allow_html=True)
                                e_c2.markdown(f'<div class="nr-metric-box"><div class="nr-metric-k">SERVER TIME</div><div class="nr-metric-v" data-nr-fin-ms="{fin_epoch_ms}">{fin_time}</div></div>', unsafe_allow_html=True)
                                e_c3.markdown(f'<div class="nr-metric-box"><div class="nr-metric-k">FINISH DAY</div><div class="nr-metric-v">{fin_day}</div></div>', unsafe_allow_html=True)
                _output_box("Training output", _filter_status_lines(st.session_state["nr_trn_log"], st.session_state["nr_trn_log"]))
                if done_t and _HAS_FRAGMENT:
                    try:
                        st.rerun(scope="app")
                    except TypeError:
                        st.rerun()

            if _HAS_FRAGMENT:
                @st.fragment(run_every=2.0 if (is_running_now or _is_task_running("index")) else None)
                def _trn_live_frag() -> None:
                    _render_trn_live_body()

                _trn_live_frag()
            else:
                _render_trn_live_body()

        with st.expander("View Generated Kaggle core.py Pipeline Commands", expanded=False):
            m_cmd_name = re.sub(r"[^a-zA-Z0-9_-]+", "-", m_clean_cur or "MyVoice")
            ds_cmd = st.session_state["nr_dataset_path"].strip() or "/kaggle/input/datasets/your-username/voice-dataset"
            st.code(
                "\\n".join([
                    "# 1. Preprocess dataset",
                    f'python core.py preprocess --model_name "{m_cmd_name}" --dataset_path "{ds_cmd}" --sample_rate {st.session_state["nr_sr"]} --cpu_cores {st.session_state["nr_cpu_cores"]} --cut_preprocess "{st.session_state["nr_audio_cutting"]}" --chunk_len {st.session_state["nr_chunk_len"]} --overlap_len {st.session_state["nr_overlap_len"]}',
                    "",
                    f'# 2. Extract features ({st.session_state["nr_f0"]} + {st.session_state["nr_embedder"]})',
                    f'python core.py extract --model_name "{m_cmd_name}" --f0_method {st.session_state["nr_f0"]} --sample_rate {st.session_state["nr_sr"]} --embedder_model {st.session_state["nr_embedder"]} --gpu {st.session_state["nr_gpu_num"]} --include_mutes {st.session_state["nr_silent_files"]}',
                    "",
                    f'# 3. Train {st.session_state["nr_vocoder"]} RVC model ({prec_val})',
                    f'python core.py train --model_name "{m_cmd_name}" --vocoder "{st.session_state["nr_vocoder"]}" --save_every_epoch {st.session_state["nr_save_every"]} --total_epoch {st.session_state["nr_total_epoch"]} --batch_size {per_gpu_bs} --sample_rate {st.session_state["nr_sr"]} --gpu {st.session_state["nr_gpu_num"]}',
                    "",
                    f'# 4. Generate {st.session_state["nr_index_algo"]} feature index',
                    f'python core.py index --model_name "{m_cmd_name}" --index_algorithm {st.session_state["nr_index_algo"]}',
                ]),
                language="bash",
            )

    # --- STEP 5: Export Model ---
    with _nr_card("nr_train_step_5"):
        _step_head(5, "Export Model", "Locate and download trained .pth checkpoints and .index retrieval files.")
        pths = _list_pth_files()
        idxs = _list_index_files()
        ex_l, ex_r = st.columns(2, gap="medium")
        with ex_l:
            sel_export_pth = st.selectbox(
                "Select Your Model (.pth)",
                options=[""] + pths,
                format_func=lambda x: "-- Select .pth checkpoint --" if x == "" else x,
                key="nr_export_pth_sel",
            )
            if sel_export_pth and (Path(now_dir) / sel_export_pth).is_file():
                p_obj = Path(now_dir) / sel_export_pth
                st.download_button(f"Download {p_obj.name}", data=_read_file_bytes_cached(str(p_obj), p_obj.stat().st_mtime), file_name=p_obj.name, type="secondary", use_container_width=True, key="nr_dl_pth_btn")
        with ex_r:
            sel_export_idx = st.selectbox(
                "Select Index File (.index)",
                options=[""] + idxs,
                format_func=lambda x: "-- Select .index file --" if x == "" else x,
                key="nr_export_idx_sel",
            )
            if sel_export_idx and (Path(now_dir) / sel_export_idx).is_file():
                i_obj = Path(now_dir) / sel_export_idx
                st.download_button(f"Download {i_obj.name}", data=_read_file_bytes_cached(str(i_obj), i_obj.stat().st_mtime), file_name=i_obj.name, type="secondary", use_container_width=True, key="nr_dl_idx_btn")

        if st.button("Refresh trained models and indexes", type="secondary", use_container_width=True, key="nr_btn_refresh_exports"):
            _clear_fs_caches()
            st.rerun()

# ==============================================================================
# TAB 2: MODELS (EXACT 3-STEP STUDIO PARITY)
# ==============================================================================
elif active_tab == "Models":
    _page_header("Models", "Fetch remote voice archives, drop local weights, and manage pretrained bases for Kaggle runs.")

    # --- STEP 1: Download Model ---
    with _nr_card("nr_models_step_1"):
        _step_head(1, "Download Model", "Add a remote archive or checkpoint URL to sync into your Samples and TTS library.")
        st.markdown('<div class="nr-field-title">Model Link</div><div class="nr-field-help">Direct URL from Hugging Face, Pixeldrain, or Google Drive.</div>', unsafe_allow_html=True)
        model_link = st.text_input("Model Link", value="", placeholder="https://huggingface.co/.../resolve/main/model.zip", label_visibility="collapsed", key="nr_model_link_inp")
        _output_box("Download output", st.session_state["nr_dl_log"])
        if st.button(_tr("Download Model"), type="primary", use_container_width=True, key="nr_btn_dl_model"):
            url_clean = model_link.strip()
            if not url_clean.startswith(("http://", "https://")):
                st.warning("Enter a valid https:// model link (Hugging Face, Pixeldrain, or Drive).")
            else:
                with st.spinner("Downloading model into library..."):
                    _, out_txt = _run_core_sync("run_download_script", {"model_link": url_clean})
                    _clear_fs_caches()
                    label = re.sub(r"[^a-zA-Z0-9_-]+", "-", Path(urllib.parse.urlsplit(url_clean).path).stem or "DownloadedModel")
                    st.session_state["nr_manifest"].append({
                        "id": f"dl-{int(time.time())}",
                        "name": label,
                        "category": "voice",
                        "url": url_clean,
                    })
                    st.session_state["nr_dl_log"] = f"✔ {out_txt}"
                st.rerun()

    # --- STEP 2: Drop Files ---
    with _nr_card("nr_models_step_2"):
        _step_head(2, "Drop Files", "Load local .pth weights, .index files, and .zip archives directly from your computer.")
        dropped = st.file_uploader(
            "Select Model Files (.pth / .index / .zip)",
            type=["pth", "index", "zip"],
            accept_multiple_files=True,
            key="nr_drop_models_up",
        )
        if dropped and st.button("Import Uploaded Model Files", type="secondary", key="nr_btn_import_dropped"):
            dl_dir = Path(now_dir) / "logs" / "downloads"
            dl_dir.mkdir(parents=True, exist_ok=True)
            saved_names: list[str] = []
            for uf in dropped:
                if uf.name.lower().endswith(".zip"):
                    tmp_z = dl_dir / uf.name
                    tmp_z.write_bytes(uf.getvalue())
                    with zipfile.ZipFile(tmp_z, "r") as zf:
                        for m in zf.namelist():
                            if not m.endswith("/") and Path(m).name.lower().endswith((".pth", ".index")):
                                dst = dl_dir / Path(m).name
                                dst.write_bytes(zf.read(m))
                                saved_names.append(dst.name)
                    tmp_z.unlink(missing_ok=True)
                else:
                    dst = dl_dir / uf.name
                    dst.write_bytes(uf.getvalue())
                    saved_names.append(dst.name)
            _clear_fs_caches()
            st.session_state["nr_dl_log"] = f"✔ Imported {', '.join(saved_names)} into logs/downloads/ -> synced to Samples & TTS."
            st.rerun()

    # --- STEP 3: Download Pretrained Models ---
    with _nr_card("nr_models_step_3"):
        _step_head(3, "Download Pretrained Models", "Queue base pretrained architectures or custom Hugging Face weights for Kaggle setup.")
        st.markdown('<div class="nr-info-box"><strong>Training recommendation.</strong> For most models, start with Auto pretrained weights and the default ContentVec embedder. Third-party pretrained weights, especially older pairs, and optional embedders can reduce final voice quality if they are a poor match for your settings or dataset.</div>', unsafe_allow_html=True)

        p_l, p_r = st.columns(2, gap="medium")
        with p_l:
            pre_arch = st.selectbox("Pretrained", ["Legacy Core 1.5 V2", "Titan", "Snowie", "KLM", "Ov2Super", "Rin_E3"], index=0, key="nr_pre_arch_sel")
        with p_r:
            sr_opts_pre = ["32k"] if pre_arch == "Legacy Core 1.5 V2" else ["32k", "40k", "48k"]
            pre_sr = st.selectbox("Sampling Rate", sr_opts_pre, index=0, key="nr_pre_sr_sel")

        use_custom_pre_m = st.checkbox("Custom Pretrains", value=bool(st.session_state["nr_custom_pre"]), key="nr_models_cpre_chk")
        cpre_g_in = ""
        cpre_d_in = ""
        if use_custom_pre_m:
            cg_l, cg_r = st.columns(2, gap="medium")
            with cg_l:
                cpre_g_in = st.text_input("Repository, pretrained ZIP or G .pth link", value=st.session_state["nr_custom_g"], placeholder="https://huggingface.co/author/pretrained", key="nr_models_cpre_g")
            with cg_r:
                cpre_d_in = st.text_input("Matching D .pth link (optional)", value=st.session_state["nr_custom_d"], placeholder="Leave empty for automatic discovery", key="nr_models_cpre_d")

        btn_pre_lbl = "Download pretrained pair" if use_custom_pre_m else ("Download Legacy Core 1.5 V2 (G + D)" if pre_arch == "Legacy Core 1.5 V2" else "Download")
        if st.button(btn_pre_lbl, type="primary", use_container_width=True, key="nr_btn_dl_pretrained"):
            cdir = Path(now_dir) / "rvc" / "models" / "pretraineds" / "custom"
            cdir.mkdir(parents=True, exist_ok=True)
            ts = time.strftime("%H:%M:%S")
            if use_custom_pre_m:
                g_val = cpre_g_in.strip()
                d_val = (cpre_d_in.strip() or _infer_matching_d_url(g_val)).strip()
                if not g_val and not d_val:
                    st.warning("Enter a Hugging Face repo, ZIP, or G .pth URL first.")
                else:
                    st.session_state["nr_custom_pre"] = True
                    lbl_pre = g_val.split("/")[-1] or "custom pretrain"
                    msg_pre = _start_pretrain_download_bg({
                        "use_custom": True,
                        "g_val": g_val,
                        "d_val": d_val,
                        "cdir": str(cdir),
                        "label": lbl_pre,
                    })
                    st.session_state["nr_pre_log_text"] = msg_pre
                    st.session_state["nr_pre_log"].append(f"[{ts}] {msg_pre}")
                    st.rerun()
            else:
                msg_pre = _start_pretrain_download_bg({
                    "use_custom": False,
                    "pre_arch": pre_arch,
                    "pre_sr": pre_sr,
                    "cdir": str(cdir),
                    "label": f"{pre_arch} ({pre_sr})",
                })
                st.session_state["nr_pre_log_text"] = msg_pre
                st.session_state["nr_pre_log"].append(f"[{ts}] {msg_pre}")
                st.rerun()

        with _nr_box("nr_pre_live_box"):
            def _render_pre_live_body() -> None:
                _, done_pre = _poll_core_tasks()
                pre_txt = st.session_state.get("nr_pre_log_text") or "\\n".join(st.session_state["nr_pre_log"][-20:])
                if pre_txt:
                    _output_box("Pretrained Download Log", pre_txt)
                if done_pre and _HAS_FRAGMENT:
                    try:
                        st.rerun(scope="app")
                    except TypeError:
                        st.rerun()

            if _HAS_FRAGMENT:
                @st.fragment(run_every=1.5 if _is_task_running("pretrained") else None)
                def _pre_live_frag() -> None:
                    _render_pre_live_body()

                _pre_live_frag()
            else:
                _render_pre_live_body()

        if st.session_state["nr_manifest"]:
            st.markdown('<div class="nr-field-title" style="margin-top:12px">Active Boot Manifest</div>', unsafe_allow_html=True)
            mh1, mh2, mh3, mh4 = st.columns([2, 1.1, 4.5, 0.7], gap="small")
            mh1.markdown('<div style="font-size:11px;font-weight:600;color:#AD8288;border-bottom:1px solid #47161D;padding-bottom:6px">Name</div>', unsafe_allow_html=True)
            mh2.markdown('<div style="font-size:11px;font-weight:600;color:#AD8288;border-bottom:1px solid #47161D;padding-bottom:6px">Type</div>', unsafe_allow_html=True)
            mh3.markdown('<div style="font-size:11px;font-weight:600;color:#AD8288;border-bottom:1px solid #47161D;padding-bottom:6px">Source URL</div>', unsafe_allow_html=True)
            mh4.markdown('<div style="font-size:11px;font-weight:600;color:#AD8288;border-bottom:1px solid #47161D;padding-bottom:6px">Action</div>', unsafe_allow_html=True)
            for idx_m, item in enumerate(list(st.session_state["nr_manifest"])):
                mc1, mc2, mc3, mc4 = st.columns([2, 1.1, 4.5, 0.7], gap="small")
                mc1.markdown(f'<div style="font-size:12px;font-weight:600;color:#FDF7F7;padding-top:6px">{html.escape(str(item.get("name","")))}</div>', unsafe_allow_html=True)
                mc2.markdown(f'<div style="font-size:12px;color:#DFBFC3;padding-top:6px">{html.escape(str(item.get("category","")))}</div>', unsafe_allow_html=True)
                mc3.markdown(f'<div style="font-family:JetBrains Mono,monospace;font-size:11px;color:#AD8288;padding-top:6px;overflow:hidden;text-overflow:ellipsis">{html.escape(str(item.get("url","")))}</div>', unsafe_allow_html=True)
                if mc4.button("✕", key=f"nr_rm_man_{idx_m}", help="Remove item"):
                    st.session_state["nr_manifest"].pop(idx_m)
                    st.rerun()
`;
}
