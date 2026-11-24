export function buildStreamlitTabsPart3(): string {
  return `
# ==============================================================================
# TAB 3: SAMPLES (EXACT 3-STEP MULTI-CLIP QUEUE & ADVANCED SETTINGS)
# ==============================================================================
elif active_tab == "Samples":
    _page_header(
        "Samples",
        "Load trained checkpoints, queue recordings, and audition pitch and index conversions.",
        show_readme=False,
    )

    cfg_now = _load_config()
    q_filter = ""
    if bool(cfg_now.get("filter", False)):
        q_filter = st.text_input("Filter models & indexes", value="", placeholder="Type to filter .pth / .index...", key="nr_inf_filter_q")

    pths = _list_pth_files(q_filter)
    idxs = _list_index_files(q_filter)
    audio_inputs = _list_audio_inputs()

    # --- STEP 1: Voice Model ---
    with _nr_card("nr_samples_step_1"):
        _step_head(1, "Voice Model", "Pick the default .pth voice checkpoint and optional FAISS .index file for new clips.")
        m_l, m_r = st.columns(2, gap="medium")
        with m_l:
            st.markdown('<div class="nr-field-title">Voice Model</div>', unsafe_allow_html=True)
            cur_pth = st.session_state.get("nr_inf_pth", "")
            pth_opts = [""] + pths
            if cur_pth and cur_pth not in pth_opts:
                pth_opts.append(cur_pth)
            sel_pth = st.selectbox(
                "Voice Model",
                options=pth_opts,
                index=pth_opts.index(cur_pth) if cur_pth in pth_opts else 0,
                format_func=lambda x: "-- Select Voice Model (.pth) --" if x == "" else x,
                label_visibility="collapsed",
                key="nr_inf_pth_sel",
            )
            if sel_pth != st.session_state.get("nr_inf_pth", ""):
                st.session_state["nr_inf_pth"] = sel_pth
                auto_idx = _match_index(sel_pth, idxs)
                if auto_idx:
                    st.session_state["nr_inf_idx"] = auto_idx
        with m_r:
            st.markdown('<div class="nr-field-title">Index File</div>', unsafe_allow_html=True)
            cur_idx = st.session_state.get("nr_inf_idx", "")
            idx_opts = [""] + idxs
            if cur_idx and cur_idx not in idx_opts:
                idx_opts.append(cur_idx)
            sel_idx = st.selectbox(
                "Index File",
                options=idx_opts,
                index=idx_opts.index(cur_idx) if cur_idx in idx_opts else 0,
                format_func=lambda x: "-- Select Index File (.index) --" if x == "" else x,
                label_visibility="collapsed",
                key="nr_inf_idx_sel",
            )
            st.session_state["nr_inf_idx"] = sel_idx

        b_un, b_rf, _ = st.columns([1.2, 1.8, 3], gap="small")
        with b_un:
            if st.button("Unload Voice", type="secondary", use_container_width=True, key="nr_btn_unload_voice"):
                st.session_state["nr_inf_pth"] = ""
                st.session_state["nr_inf_idx"] = ""
                st.session_state["nr_inf_log"] = "Voice model unloaded from VRAM."
                st.rerun()
        with b_rf:
            if st.button("Refresh models and indexes", type="secondary", use_container_width=True, key="nr_btn_ref_models"):
                _clear_fs_caches()
                st.rerun()

    # --- STEP 2: Audio Source & Multi-Sample Queue ---
    with _nr_card("nr_samples_step_2"):
        s2_l, s2_r = st.columns([3.5, 1.5], gap="small")
        with s2_l:
            _step_head(2, "Audio Source", "Drop or select one or multiple recordings, customize Pitch & Index per sample, or point to a Kaggle folder.")
        with s2_r:
            st.session_state.setdefault("nr_inf_mode", "Audio Files")
            m1, m2 = st.columns(2, gap="small")
            if m1.button("Audio Files", type="primary" if st.session_state["nr_inf_mode"] == "Audio Files" else "secondary", use_container_width=True, key="nr_mode_files"):
                st.session_state["nr_inf_mode"] = "Audio Files"
                st.rerun()
            if m2.button("Batch", type="primary" if st.session_state["nr_inf_mode"] == "Batch" else "secondary", use_container_width=True, key="nr_mode_batch"):
                st.session_state["nr_inf_mode"] = "Batch"
                st.rerun()

        if st.session_state["nr_inf_mode"] == "Audio Files":
            up_clips = st.file_uploader(
                "Drop Audio Here — or — Click to Upload (WAV, MP3, FLAC, OGG)",
                type=["wav", "mp3", "flac", "ogg", "m4a"],
                accept_multiple_files=True,
                key="nr_inf_up",
            )
            if up_clips:
                au_dir = Path(now_dir) / "assets" / "audios"
                au_dir.mkdir(parents=True, exist_ok=True)
                queued_paths = {c["path"] for c in st.session_state["nr_sample_queue"]}
                added_n = 0
                for uf in up_clips:
                    dst = au_dir / uf.name
                    if not dst.is_file():
                        dst.write_bytes(uf.getvalue())
                    rel_dst = str(dst.relative_to(now_dir))
                    if rel_dst not in queued_paths:
                        st.session_state["nr_sample_queue"].append({
                            "id": f"clip-{int(time.time()*1000)}-{added_n}",
                            "name": uf.name,
                            "path": rel_dst,
                            "pitch": int(st.session_state.get("nr_def_pitch", 0)),
                            "index_rate": float(st.session_state.get("nr_def_idx_rate", 0.75)),
                            "protect": float(st.session_state.get("nr_def_protect", 0.5)),
                            "volume_envelope": float(st.session_state.get("nr_def_vol_env", 1.0)),
                            "autotune": bool(st.session_state.get("nr_def_autotune", False)),
                            "proposed_pitch": bool(st.session_state.get("nr_def_proposed", False)),
                            "split_audio": bool(st.session_state.get("nr_def_split", False)),
                            "format": str(st.session_state.get("nr_def_fmt", "WAV")),
                            "model": "",
                            "index": "",
                            "converted_path": "",
                        })
                        queued_paths.add(rel_dst)
                        added_n += 1
                if added_n > 0:
                    st.session_state["nr_inf_log"] = f"Added {added_n} uploaded clip(s) to sample queue."

            if audio_inputs:
                ex_l, ex_r = st.columns([3.5, 1.2], gap="small")
                with ex_l:
                    pick_au = st.selectbox(
                        "Pick existing recording from assets/audios or free_samples",
                        options=[""] + audio_inputs,
                        format_func=lambda x: "Add existing clip from library..." if x == "" else x,
                        key="nr_pick_existing_au",
                    )
                with ex_r:
                    st.markdown('<div style="height:26px"></div>', unsafe_allow_html=True)
                    if st.button("Add Clip", type="secondary", use_container_width=True, key="nr_btn_add_lib_clip") and pick_au:
                        st.session_state["nr_sample_queue"].append({
                            "id": f"clip-{int(time.time()*1000)}",
                            "name": Path(pick_au).name,
                            "path": pick_au,
                            "pitch": int(st.session_state.get("nr_def_pitch", 0)),
                            "index_rate": float(st.session_state.get("nr_def_idx_rate", 0.75)),
                            "protect": float(st.session_state.get("nr_def_protect", 0.5)),
                            "volume_envelope": float(st.session_state.get("nr_def_vol_env", 1.0)),
                            "autotune": bool(st.session_state.get("nr_def_autotune", False)),
                            "proposed_pitch": bool(st.session_state.get("nr_def_proposed", False)),
                            "split_audio": bool(st.session_state.get("nr_def_split", False)),
                            "format": str(st.session_state.get("nr_def_fmt", "WAV")),
                            "model": "",
                            "index": "",
                            "converted_path": "",
                        })
                        st.rerun()

            q_items = st.session_state["nr_sample_queue"]
            if q_items:
                qh_l, qh_m, qh_r = st.columns([3, 1.8, 1], gap="small")
                qh_l.markdown(f'<div class="nr-field-title" style="padding-top:8px">Queued Audio Samples ({len(q_items)})</div>', unsafe_allow_html=True)
                if len(q_items) > 1 and qh_m.button("Apply Default Pitch/Index to All", type="secondary", use_container_width=True, key="nr_apply_def_all"):
                    for c in q_items:
                        c["pitch"] = int(st.session_state.get("nr_def_pitch", 0))
                        c["index_rate"] = float(st.session_state.get("nr_def_idx_rate", 0.75))
                    st.rerun()
                if qh_r.button("Clear All", type="secondary", use_container_width=True, key="nr_clear_all_clips"):
                    st.session_state["nr_sample_queue"] = []
                    st.rerun()

                for idx_c, clip in enumerate(list(q_items)):
                    with st.container(border=True):
                        ch1, ch2, ch3 = st.columns([4, 1.3, 0.55], gap="small")
                        p_str = f'+{clip["pitch"]}' if clip["pitch"] > 0 else str(clip["pitch"])
                        status_tag = " · Converted" if clip.get("converted_path") else ""
                        ch1.markdown(
                            f'<div style="font-family:JetBrains Mono,monospace;font-size:13px;font-weight:600;color:#FDF7F7">{idx_c + 1}. {html.escape(clip["name"])}</div>'
                            f'<div style="font-size:12.5px;color:#C89EA4">Pitch {p_str} · Index {clip["index_rate"]:.2f} · {clip["format"].upper()}{status_tag}</div>',
                            unsafe_allow_html=True,
                        )
                        if ch2.button("Re-Convert" if clip.get("converted_path") else "Convert", type="primary", use_container_width=True, key=f"nr_conv_one_{clip['id']}"):
                            chosen_pth = clip.get("model") or st.session_state.get("nr_inf_pth", "")
                            if not chosen_pth:
                                st.warning("Select a Voice Model (.pth) in Step 1 first.")
                            else:
                                chosen_idx = clip.get("index") or st.session_state.get("nr_inf_idx", "")
                                fmt_ext = clip.get("format", "WAV").lower()
                                out_name = f"{Path(clip['name']).stem}_p{p_str}.{fmt_ext}"
                                out_abs = str(Path(now_dir) / "assets" / "audios" / out_name)
                                with st.spinner(f"Converting {clip['name']}..."):
                                    rc, msg = _run_core_sync(
                                        "run_infer_script",
                                        {
                                            "pitch": int(clip["pitch"]),
                                            "index_rate": float(clip["index_rate"]),
                                            "volume_envelope": float(clip["volume_envelope"]),
                                            "protect": float(clip["protect"]),
                                            "f0_method": st.session_state.get("nr_def_f0", "rmvpe"),
                                            "input_path": str(Path(now_dir) / clip["path"]),
                                            "output_path": out_abs,
                                            "pth_path": str(Path(now_dir) / chosen_pth),
                                            "index_path": str(Path(now_dir) / chosen_idx) if chosen_idx else "",
                                            "split_audio": bool(clip["split_audio"]),
                                            "f0_autotune": bool(clip["autotune"]),
                                            "f0_autotune_strength": 1.0,
                                            "proposed_pitch": bool(clip["proposed_pitch"]),
                                            "proposed_pitch_threshold": 155.0,
                                            "clean_audio": bool(st.session_state.get("nr_def_clean", False)),
                                            "clean_strength": float(st.session_state.get("nr_def_clean_db", 12)) / 30.0,
                                            "export_format": clip.get("format", "WAV").upper(),
                                            "embedder_model": st.session_state.get("nr_def_emb", "contentvec"),
                                            "sid": int(st.session_state.get("nr_def_sid", 0)),
                                            "formant_shifting": bool(st.session_state.get("nr_def_formant", False)),
                                            "formant_qfrency": float(st.session_state.get("nr_def_qf", 1.0)),
                                            "formant_timbre": float(st.session_state.get("nr_def_tf", 1.0)),
                                        },
                                    )
                                    st.session_state["nr_inf_log"] = msg
                                    if rc == 0 and os.path.isfile(out_abs):
                                        clip["converted_path"] = out_abs
                                        st.session_state["nr_saved_samples"].insert(0, {
                                            "id": f"out-{int(time.time()*1000)}",
                                            "name": out_name,
                                            "path": out_abs,
                                            "pitch": int(clip["pitch"]),
                                            "index_rate": float(clip["index_rate"]),
                                            "model": Path(chosen_pth).name,
                                            "created_at": time.strftime("%H:%M:%S"),
                                        })
                                st.rerun()
                        if ch3.button("✕", key=f"nr_rm_clip_{clip['id']}", help="Discard clip"):
                            st.session_state["nr_sample_queue"].pop(idx_c)
                            st.rerun()

                        src_abs = Path(now_dir) / clip["path"]
                        if src_abs.is_file():
                            st.audio(str(src_abs))

                        cp1, cp2 = st.columns(2, gap="medium")
                        with cp1:
                            clip["pitch"] = st.slider("Pitch", -32, 32, int(clip["pitch"]), step=1, key=f"nr_cp_{clip['id']}")
                        with cp2:
                            clip["index_rate"] = st.slider("Index Rate", 0.0, 1.0, float(clip["index_rate"]), step=0.05, key=f"nr_ci_{clip['id']}")

                        with st.expander("Customize model, protect, format & advanced", expanded=False):
                            ov1, ov2, ov3 = st.columns(3, gap="small")
                            clip["model"] = ov1.selectbox("Clip Voice Model", options=[""] + pths, format_func=lambda x: "Use Default (Step 1)" if x == "" else x, key=f"nr_cm_{clip['id']}")
                            clip["index"] = ov2.selectbox("Clip Index File", options=[""] + idxs, format_func=lambda x: "Use Default / None" if x == "" else x, key=f"nr_cx_{clip['id']}")
                            clip["format"] = ov3.selectbox("Export Format", ["WAV", "MP3", "FLAC"], index=["WAV", "MP3", "FLAC"].index(clip["format"].upper()) if clip["format"].upper() in ("WAV", "MP3", "FLAC") else 0, key=f"nr_cf_{clip['id']}")
                            ov4, ov5 = st.columns(2, gap="small")
                            clip["protect"] = ov4.slider("Protect", 0.0, 0.5, float(clip["protect"]), step=0.01, key=f"nr_cprot_{clip['id']}")
                            clip["volume_envelope"] = ov5.slider("Volume Envelope", 0.0, 1.0, float(clip["volume_envelope"]), step=0.05, key=f"nr_cvol_{clip['id']}")
                            t1, t2, t3 = st.columns(3, gap="small")
                            clip["autotune"] = t1.checkbox("Autotune", value=bool(clip["autotune"]), key=f"nr_cat_{clip['id']}")
                            clip["proposed_pitch"] = t2.checkbox("Proposed Pitch", value=bool(clip["proposed_pitch"]), key=f"nr_cpp_{clip['id']}")
                            clip["split_audio"] = t3.checkbox("Split", value=bool(clip["split_audio"]), key=f"nr_csp_{clip['id']}")

                        if clip.get("converted_path") and os.path.isfile(clip["converted_path"]):
                            st.markdown(f'<div style="font-size:12.5px;font-weight:600;color:#FDF7F7;margin-top:6px">Converted Output ({html.escape(Path(clip["converted_path"]).name)})</div>', unsafe_allow_html=True)
                            st.audio(clip["converted_path"])
                            st.download_button("Save Converted Clip", data=Path(clip["converted_path"]).read_bytes(), file_name=Path(clip["converted_path"]).name, type="secondary", key=f"nr_dl_c_{clip['id']}")
        else:
            st.session_state["nr_batch_folder"] = st.text_input(
                "Input Folder Path (Batch)",
                value=st.session_state.get("nr_batch_folder", "/kaggle/working/audio_inputs"),
                key="nr_batch_folder_inp",
            )

    # --- STEP 3: Conversion Parameters ---
    with _nr_card("nr_samples_step_3"):
        _step_head(3, "Conversion Parameters", "Set default Pitch, Index Rate, and export options for incoming clips or batch folders.")
        st.session_state.setdefault("nr_def_pitch", 0)
        st.session_state["nr_def_pitch"] = _synced_slider_num(
            "Default Pitch",
            "Shift the vocal register up or down in semitones between -32 and +32.",
            "nr_def_pitch",
            -32,
            32,
            0,
        )

        f0_cur = st.session_state.get("nr_def_f0", "rmvpe")
        idx_cur = float(st.session_state.get("nr_def_idx_rate", 0.75))
        env_cur = float(st.session_state.get("nr_def_vol_env", 1.0))
        prot_cur = float(st.session_state.get("nr_def_protect", 0.5))
        adv_lbl = f"Advanced Settings ({f0_cur} · index {idx_cur:.2f} · env {env_cur:g} · protect {prot_cur:g})"
        with st.expander(adv_lbl, expanded=False):
            with st.expander("Voice Matching", expanded=True):
                st.session_state["nr_def_idx_rate"] = st.slider("Search Feature Ratio", 0.0, 1.0, float(st.session_state.get("nr_def_idx_rate", 0.75)), step=0.05, key="nr_adv_idx")
                st.session_state["nr_def_vol_env"] = st.slider("Volume Envelope", 0.0, 1.0, float(st.session_state.get("nr_def_vol_env", 1.0)), step=0.05, key="nr_adv_vol")
                st.session_state["nr_def_protect"] = st.slider("Protect Voiceless Consonants", 0.0, 0.5, float(st.session_state.get("nr_def_protect", 0.5)), step=0.01, key="nr_adv_prot")

            with st.expander("Models & Algorithms", expanded=False):
                ma1, ma2, ma3 = st.columns(3, gap="small")
                st.session_state["nr_def_sid"] = ma1.number_input("Speaker ID", 0, 100, int(st.session_state.get("nr_def_sid", 0)), step=1, key="nr_adv_sid")
                st.session_state["nr_def_f0"] = ma2.selectbox("Pitch extraction algorithm", ["rmvpe", "fcpe", "swift-f0", "crepe"], key="nr_adv_f0")
                st.session_state["nr_def_emb"] = ma3.selectbox("Embedder Model", ["contentvec", "spin", "chinese-hubert-base"], key="nr_adv_emb")

            with st.expander("Advanced Pitch & Tuning", expanded=False):
                pt1, pt2, pt3 = st.columns(3, gap="small")
                st.session_state["nr_def_autotune"] = pt1.checkbox("Autotune", value=bool(st.session_state.get("nr_def_autotune", False)), key="nr_adv_at")
                st.session_state["nr_def_proposed"] = pt2.checkbox("Proposed Pitch", value=bool(st.session_state.get("nr_def_proposed", False)), key="nr_adv_pp")
                st.session_state["nr_def_split"] = pt3.checkbox("Split Audio", value=bool(st.session_state.get("nr_def_split", False)), key="nr_adv_sp")

            with st.expander("Audio Cleanup", expanded=False):
                st.session_state["nr_def_clean"] = st.checkbox("Audio cleanup (balanced rumble filter & gentle de-noise)", value=bool(st.session_state.get("nr_def_clean", False)), key="nr_adv_clean")
                st.session_state["nr_def_clean_db"] = st.slider("Noise reduction limit (dB)", 1, 30, int(st.session_state.get("nr_def_clean_db", 12)), step=1, disabled=not bool(st.session_state["nr_def_clean"]), key="nr_adv_cleandb")

            with st.expander("Formant Shifting", expanded=False):
                st.session_state["nr_def_formant"] = st.checkbox("Enable Formant Shifting", value=bool(st.session_state.get("nr_def_formant", False)), key="nr_adv_fmt_on")
                if st.session_state["nr_def_formant"]:
                    fs1, fs2 = st.columns(2, gap="small")
                    st.session_state["nr_def_qf"] = fs1.slider("Formant Quefrency", 0.0, 16.0, float(st.session_state.get("nr_def_qf", 1.0)), step=0.1, key="nr_adv_qf")
                    st.session_state["nr_def_tf"] = fs2.slider("Formant Timbre", 0.0, 16.0, float(st.session_state.get("nr_def_tf", 1.0)), step=0.1, key="nr_adv_tf")

            with st.expander("Output Settings", expanded=False):
                os1, os2 = st.columns([3, 1.2], gap="small")
                st.session_state["nr_def_out_path"] = os1.text_input("Output Path", value=st.session_state.get("nr_def_out_path", "assets/audios/output.wav"), key="nr_adv_outp")
                st.session_state["nr_def_fmt"] = os2.selectbox("Export Format", ["WAV", "MP3", "FLAC"], key="nr_adv_fmt")
                if st.button("Clear Outputs (Deletes all audios in assets/audios)", type="secondary", use_container_width=True, key="nr_btn_clr_out"):
                    au_dir = Path(now_dir) / "assets" / "audios"
                    if au_dir.is_dir():
                        for f in au_dir.iterdir():
                            if f.is_file():
                                f.unlink(missing_ok=True)
                    st.session_state["nr_saved_samples"] = []
                    st.session_state["nr_inf_log"] = "Cleared all generated outputs in assets/audios."
                    st.rerun()

        with st.container(border=True):
            inf_terms = st.checkbox("I agree to the terms of use — Confirm responsible usage guidelines prior to rendering voice conversion samples.", value=True, key="nr_inf_terms")
        q_len = len(st.session_state["nr_sample_queue"])
        conv_btn_lbl = f"Convert All ({q_len} Samples)" if (st.session_state["nr_inf_mode"] == "Audio Files" and q_len > 1) else _tr("Convert")
        if st.button(conv_btn_lbl, type="primary", use_container_width=True, key="nr_btn_conv_all"):
            pth_sel = st.session_state.get("nr_inf_pth", "")
            idx_sel = st.session_state.get("nr_inf_idx", "")
            if not inf_terms:
                st.warning("Confirm the terms of use checkbox first.")
            elif not pth_sel:
                st.warning("Select a Voice Model (.pth) in Step 1 first.")
            elif st.session_state["nr_inf_mode"] == "Batch":
                with st.spinner("Running batch folder conversion..."):
                    rc, msg = _run_core_sync(
                        "run_batch_infer_script",
                        {
                            "pitch": int(st.session_state.get("nr_def_pitch", 0)),
                            "index_rate": float(st.session_state.get("nr_def_idx_rate", 0.75)),
                            "volume_envelope": float(st.session_state.get("nr_def_vol_env", 1.0)),
                            "protect": float(st.session_state.get("nr_def_protect", 0.5)),
                            "f0_method": st.session_state.get("nr_def_f0", "rmvpe"),
                            "input_folder": st.session_state.get("nr_batch_folder", "/kaggle/working/audio_inputs"),
                            "output_folder": str(Path(now_dir) / "assets" / "audios"),
                            "pth_path": str(Path(now_dir) / pth_sel),
                            "index_path": str(Path(now_dir) / idx_sel) if idx_sel else "",
                            "export_format": st.session_state.get("nr_def_fmt", "WAV"),
                            "embedder_model": st.session_state.get("nr_def_emb", "contentvec"),
                            "sid": int(st.session_state.get("nr_def_sid", 0)),
                        },
                    )
                    st.session_state["nr_inf_log"] = msg
                st.rerun()
            elif not q_len:
                st.warning("Upload or add audio files in Step 2 first.")
            else:
                with st.spinner(f"Converting {q_len} sample(s)..."):
                    msgs = []
                    for clip in st.session_state["nr_sample_queue"]:
                        c_pth = clip.get("model") or pth_sel
                        c_idx = clip.get("index") or idx_sel
                        p_str = f'+{clip["pitch"]}' if clip["pitch"] > 0 else str(clip["pitch"])
                        out_name = f"{Path(clip['name']).stem}_p{p_str}.{clip.get('format', 'WAV').lower()}"
                        out_abs = str(Path(now_dir) / "assets" / "audios" / out_name)
                        rc, msg = _run_core_sync(
                            "run_infer_script",
                            {
                                "pitch": int(clip["pitch"]),
                                "index_rate": float(clip["index_rate"]),
                                "volume_envelope": float(clip["volume_envelope"]),
                                "protect": float(clip["protect"]),
                                "f0_method": st.session_state.get("nr_def_f0", "rmvpe"),
                                "input_path": str(Path(now_dir) / clip["path"]),
                                "output_path": out_abs,
                                "pth_path": str(Path(now_dir) / c_pth),
                                "index_path": str(Path(now_dir) / c_idx) if c_idx else "",
                                "split_audio": bool(clip["split_audio"]),
                                "f0_autotune": bool(clip["autotune"]),
                                "f0_autotune_strength": 1.0,
                                "proposed_pitch": bool(clip["proposed_pitch"]),
                                "proposed_pitch_threshold": 155.0,
                                "clean_audio": bool(st.session_state.get("nr_def_clean", False)),
                                "clean_strength": float(st.session_state.get("nr_def_clean_db", 12)) / 30.0,
                                "export_format": clip.get("format", "WAV").upper(),
                                "embedder_model": st.session_state.get("nr_def_emb", "contentvec"),
                                "sid": int(st.session_state.get("nr_def_sid", 0)),
                                "formant_shifting": bool(st.session_state.get("nr_def_formant", False)),
                                "formant_qfrency": float(st.session_state.get("nr_def_qf", 1.0)),
                                "formant_timbre": float(st.session_state.get("nr_def_tf", 1.0)),
                            },
                        )
                        msgs.append(f"{clip['name']}: {msg}")
                        if rc == 0 and os.path.isfile(out_abs):
                            clip["converted_path"] = out_abs
                            st.session_state["nr_saved_samples"].insert(0, {
                                "id": f"out-{int(time.time()*1000)}-{clip['id']}",
                                "name": out_name,
                                "path": out_abs,
                                "pitch": int(clip["pitch"]),
                                "index_rate": float(clip["index_rate"]),
                                "model": Path(c_pth).name,
                                "created_at": time.strftime("%H:%M:%S"),
                            })
                    st.session_state["nr_inf_log"] = "\\n".join(msgs)
                st.rerun()

        _output_box("Sample output", st.session_state["nr_inf_log"])

        cli_pth = st.session_state.get("nr_inf_pth", "") or "logs/Model/Model.pth"
        cli_idx = st.session_state.get("nr_inf_idx", "") or "logs/Model/Model.index"
        first_clip = st.session_state["nr_sample_queue"][0] if st.session_state["nr_sample_queue"] else None
        cli_in = first_clip["path"] if first_clip else (st.session_state.get("nr_batch_folder", "/kaggle/working/audio_inputs") if st.session_state["nr_inf_mode"] == "Batch" else "sample_input.wav")
        cli_out = st.session_state.get("nr_def_out_path", "assets/audios/output.wav")
        cli_pitch = int(first_clip["pitch"]) if first_clip else int(st.session_state.get("nr_def_pitch", 0))
        cli_idx_r = float(first_clip["index_rate"]) if first_clip else idx_cur
        cli_prot = float(first_clip["protect"]) if first_clip else prot_cur
        cli_env = float(first_clip["volume_envelope"]) if first_clip else env_cur
        cli_at = bool(first_clip["autotune"]) if first_clip else bool(st.session_state.get("nr_def_autotune", False))
        cli_pp = bool(first_clip["proposed_pitch"]) if first_clip else bool(st.session_state.get("nr_def_proposed", False))
        cli_sp = bool(first_clip["split_audio"]) if first_clip else bool(st.session_state.get("nr_def_split", False))
        cli_fmt = str(first_clip["format"] if first_clip else st.session_state.get("nr_def_fmt", "WAV")).upper()
        cli_cmd = (" " + chr(92) + chr(10)).join([
            "python core.py infer",
            f'  --pth_path "{cli_pth}"',
            f'  --index_path "{cli_idx}"',
            f'  --input_path "{cli_in}"',
            f'  --output_path "{cli_out}"',
            f'  --sid {int(st.session_state.get("nr_def_sid", 0))} --f0_method {f0_cur} --embedder_model {st.session_state.get("nr_def_emb", "contentvec")}',
            f'  --pitch {cli_pitch} --index_rate {cli_idx_r:.2f} --protect {cli_prot:g} --volume_envelope {cli_env:g}',
            f'  --f0_autotune {cli_at} --proposed_pitch {cli_pp} --split_audio {cli_sp}',
            f'  --export_format {cli_fmt}',
        ])
        st.code(cli_cmd, language="bash")

        saved_list = st.session_state.get("nr_saved_samples", [])
        if saved_list:
            with st.container(border=True):
                st.markdown(f'<div class="nr-field-title">Recent Generated Samples ({len(saved_list)})</div>', unsafe_allow_html=True)
                for idx_s, s_item in enumerate(list(saved_list[:12])):
                    sr1, sr2, sr3 = st.columns([4, 1.1, 0.5], gap="small")
                    sp_str = f'+{s_item["pitch"]}' if s_item["pitch"] > 0 else str(s_item["pitch"])
                    sr1.markdown(
                        f'<div style="font-family:JetBrains Mono,monospace;font-size:12px;color:#FDF7F7">{html.escape(s_item["name"])}</div>'
                        f'<div style="font-size:11px;color:#AD8288">{html.escape(s_item["model"])} · Pitch {sp_str} · idx {s_item["index_rate"]:.2f} · {s_item["created_at"]}</div>',
                        unsafe_allow_html=True,
                    )
                    if os.path.isfile(s_item["path"]):
                        sr2.download_button("Save", data=Path(s_item["path"]).read_bytes(), file_name=s_item["name"], type="secondary", use_container_width=True, key=f"nr_dl_sav_{idx_s}")
                        st.audio(s_item["path"])
                    if sr3.button("✕", key=f"nr_rm_sav_{idx_s}"):
                        st.session_state["nr_saved_samples"].pop(idx_s)
                        st.rerun()

# ==============================================================================
# TAB 4: FREE SAMPLES (6 DRY MONO 32k WAV CLIPS + 3-COLUMN CARDS)
# ==============================================================================
elif active_tab == "Free Samples":
    fh1, fh2, fh3 = st.columns([3.4, 1.35, 1.35], gap="small")
    with fh1:
        _page_header("Free Samples", "Ready-to-use dry mono talking recordings (3 male, 3 female) to preview, download for Kaggle, or load straight into Samples.", show_readme=False)
    fs_dir = Path(now_dir) / "assets" / "free_samples"
    fs_dir.mkdir(parents=True, exist_ok=True)
    if fh2.button(_tr("Load All 6 in Samples"), type="primary", use_container_width=True, key="nr_fs_load_all"):
        queued_paths = {c["path"] for c in st.session_state["nr_sample_queue"]}
        for item in FREE_SAMPLES_CATALOG:
            local_p = fs_dir / item["repoWav"]
            if not local_p.is_file():
                try:
                    urllib.request.urlretrieve(f"https://raw.githubusercontent.com/HunterThatsMe/NoRegretsRVC/main/public/assets/aistudio/{item['repoWav']}", local_p)
                except Exception:
                    pass
            rel_p = str(local_p.relative_to(now_dir))
            if rel_p not in queued_paths:
                st.session_state["nr_sample_queue"].append({
                    "id": f"fs-{item['id']}-{int(time.time()*1000)}",
                    "name": item["fileName"],
                    "path": rel_p,
                    "pitch": int(st.session_state.get("nr_def_pitch", 0)),
                    "index_rate": float(st.session_state.get("nr_def_idx_rate", 0.75)),
                    "protect": 0.5,
                    "volume_envelope": 1.0,
                    "autotune": False,
                    "proposed_pitch": False,
                    "split_audio": False,
                    "format": "WAV",
                    "model": "",
                    "index": "",
                    "converted_path": "",
                })
        st.session_state["nr_active_tab"] = "Samples"
        st.rerun()
    with fh3:
        zbuf = io.BytesIO()
        available_wavs = [fs_dir / item["repoWav"] for item in FREE_SAMPLES_CATALOG if (fs_dir / item["repoWav"]).is_file()]
        if available_wavs:
            with zipfile.ZipFile(zbuf, "w", zipfile.ZIP_DEFLATED) as zf:
                for p in available_wavs:
                    zf.write(p, f"NoRegretsRVC_Free_Samples/{p.name}")
            st.download_button(_tr("Download All 6 (.zip)"), data=zbuf.getvalue(), file_name="NoRegretsRVC_Free_Samples.zip", type="secondary", use_container_width=True, key="nr_fs_zip_btn")
        elif st.button(_tr("Download All 6 (.zip)"), type="secondary", use_container_width=True, key="nr_fs_fetch_all"):
            for item in FREE_SAMPLES_CATALOG:
                local_p = fs_dir / item["repoWav"]
                if not local_p.is_file():
                    try:
                        urllib.request.urlretrieve(f"https://raw.githubusercontent.com/HunterThatsMe/NoRegretsRVC/main/public/assets/aistudio/{item['repoWav']}", local_p)
                    except Exception:
                        pass
            st.rerun()

    with _nr_card("nr_free_filter_bar"):
        fb_l, fb_1, fb_2, fb_3 = st.columns([4, 1, 1, 1], gap="small")
        fb_l.markdown('<div style="font-size:13.5px;color:#DFBFC3;padding-top:6px">Clean, dry mono WAVs (32 kHz) with zero background music or reverb — ideal for RVC conversion tests.</div>', unsafe_allow_html=True)
        for col_f, g_lbl, g_val in ((fb_1, "All (6)", "All"), (fb_2, "Male (3)", "Male"), (fb_3, "Female (3)", "Female")):
            if col_f.button(g_lbl, type="primary" if st.session_state["nr_free_gender"] == g_val else "secondary", use_container_width=True, key=f"nr_fs_flt_{g_val}"):
                st.session_state["nr_free_gender"] = g_val
                st.rerun()

    groups = []
    if st.session_state["nr_free_gender"] in ("All", "Male"):
        groups.append(("Male Speaking Samples (3)", "Dry mono 32 kHz WAV recordings across narration, voiceover, and casual dialogue.", [x for x in FREE_SAMPLES_CATALOG if x["gender"] == "male"]))
    if st.session_state["nr_free_gender"] in ("All", "Female"):
        groups.append(("Female Speaking Samples (3)", "Dry mono 32 kHz WAV recordings across calm support, clear reading, and assistant speech.", [x for x in FREE_SAMPLES_CATALOG if x["gender"] == "female"]))

    for g_idx, (g_title, g_sub, items) in enumerate(groups):
        with _nr_card(f"nr_free_group_{g_idx}"):
            st.markdown(f'<div class="nr-step-title">{html.escape(g_title)}</div><div class="nr-step-desc" style="margin-bottom:12px">{html.escape(g_sub)}</div>', unsafe_allow_html=True)
            cols = st.columns(3, gap="medium")
            for idx_g, sample in enumerate(items):
                with cols[idx_g % 3]:
                    with st.container(border=True):
                        st.markdown(
                            f'<div style="display:flex;justify-content:space-between;align-items:flex-start;gap:8px">'
                            f'<div><div style="font-weight:600;font-size:15px;color:#FDF7F7">{html.escape(sample["title"])}</div>'
                            f'<div style="font-size:12.5px;color:#C89EA4">{html.escape(sample["style"])} · {html.escape(sample["durationLabel"])}</div></div>'
                            f'<span class="nr-mono-badge">Mono 32k WAV</span></div>'
                            f'<div class="nr-quote-box">“{html.escape(sample["transcript"])}”</div>',
                            unsafe_allow_html=True,
                        )
                        local_wav = fs_dir / sample["repoWav"]
                        if local_wav.is_file():
                            st.audio(str(local_wav))
                        cb1, cb2 = st.columns(2, gap="small")
                        if cb1.button("Use in Samples", type="primary", use_container_width=True, key=f"nr_fs_use_{sample['id']}"):
                            if not local_wav.is_file():
                                try:
                                    urllib.request.urlretrieve(f"https://raw.githubusercontent.com/HunterThatsMe/NoRegretsRVC/main/public/assets/aistudio/{sample['repoWav']}", local_wav)
                                except Exception:
                                    pass
                            st.session_state["nr_sample_queue"].append({
                                "id": f"fs-{sample['id']}-{int(time.time()*1000)}",
                                "name": sample["fileName"],
                                "path": str(local_wav.relative_to(now_dir)),
                                "pitch": int(st.session_state.get("nr_def_pitch", 0)),
                                "index_rate": float(st.session_state.get("nr_def_idx_rate", 0.75)),
                                "protect": 0.5,
                                "volume_envelope": 1.0,
                                "autotune": False,
                                "proposed_pitch": False,
                                "split_audio": False,
                                "format": "WAV",
                                "model": "",
                                "index": "",
                                "converted_path": "",
                            })
                            st.session_state["nr_active_tab"] = "Samples"
                            st.rerun()
                        if local_wav.is_file():
                            cb2.download_button("Download", data=local_wav.read_bytes(), file_name=sample["fileName"], type="secondary", use_container_width=True, key=f"nr_fs_dl_{sample['id']}")
                        else:
                            if cb2.button("Fetch WAV", type="secondary", use_container_width=True, key=f"nr_fs_ft_{sample['id']}"):
                                try:
                                    urllib.request.urlretrieve(f"https://raw.githubusercontent.com/HunterThatsMe/NoRegretsRVC/main/public/assets/aistudio/{sample['repoWav']}", local_wav)
                                except Exception:
                                    pass
                                st.rerun()

# ==============================================================================
# TAB 5: DIAGNOSTICS (REAL-DATA BEST EPOCH ANALYZER & MEL LOSS GRAPH)
# ==============================================================================
elif active_tab == "Diagnostics":
    _page_header("Diagnostics", "Model-specific Best Epoch finder driven by your actual training mel loss, dataset profile, and listening feedback.")

    m_diag_raw = st.session_state["nr_model_name"].strip()
    m_diag_name = m_diag_raw or "Untitled Model"
    auto_log = _read_train_log(m_diag_raw)
    raw_source = st.session_state.get("nr_diag_log_input", "").strip() or auto_log
    parsed_diag = _parse_training_log(raw_source) if raw_source else None
    ds_slices_d = _dataset_slice_count(m_diag_raw) if m_diag_raw else 0

    eff_batch_d = int(st.session_state["nr_batch_size"]) * max(1, len([x for x in str(st.session_state["nr_gpu_num"]).split("-") if x.strip()]))
    sr_d = int(st.session_state["nr_sr"])
    sev_d = int(st.session_state["nr_save_every"])
    tot_d = int(st.session_state["nr_total_epoch"])
    voc_d = str(st.session_state["nr_vocoder"])

    zone = _analyze_zones(
        parsed_diag,
        sev_d,
        tot_d,
        sr_d,
        st.session_state["nr_diag_sound"],
        dataset_slices=ds_slices_d,
        vocoder=voc_d,
        model_name=m_diag_raw,
    )

    with _nr_card("nr_diag_main"):
        st.markdown(
            f'<div class="nr-step-head"><span class="nr-step-num" style="background:#200B0E;border:1px solid rgba(229,25,35,0.5);color:#E51923">⌖</span>'
            f'<div><div class="nr-step-title">Best Epoch Analyzer <span class="nr-mono-badge" style="margin-left:6px">Save Every {sev_d} Epochs</span></div>'
            f'<div class="nr-step-desc">Identifies the best saved checkpoint epoch from real training logs, mel loss convergence, dataset size, and your sound audition checks.</div></div></div>',
            unsafe_allow_html=True,
        )

        # Active Model Reference Bar (6 cards)
        m1, m2, m3, m4, m5, m6 = st.columns(6, gap="small")
        m1.markdown(f'<div class="nr-metric-box"><div class="nr-metric-k">Active Model</div><div class="nr-metric-v">{html.escape(m_diag_name)}</div></div>', unsafe_allow_html=True)
        m2.markdown(f'<div class="nr-metric-box"><div class="nr-metric-k">Vocoder</div><div class="nr-metric-v">{html.escape(voc_d)}</div></div>', unsafe_allow_html=True)
        m3.markdown(f'<div class="nr-metric-box"><div class="nr-metric-k">Sample Rate</div><div class="nr-metric-v">{sr_d} Hz</div></div>', unsafe_allow_html=True)
        m4.markdown(f'<div class="nr-metric-box"><div class="nr-metric-k">Effective Batch</div><div class="nr-metric-v">{eff_batch_d}</div></div>', unsafe_allow_html=True)
        m5.markdown(f'<div class="nr-metric-box"><div class="nr-metric-k">Save Interval</div><div class="nr-metric-v" style="color:#E51923">Every {sev_d} ep</div></div>', unsafe_allow_html=True)
        m6.markdown(f'<div class="nr-metric-box"><div class="nr-metric-k">Dataset Clips</div><div class="nr-metric-v">{ds_slices_d if ds_slices_d > 0 else "—"}</div></div>', unsafe_allow_html=True)

        if zone["confidence"] == "untrained":
            with st.container(border=True):
                st.markdown(
                    f'<div style="padding:14px 6px;text-align:center">'
                    f'<div style="font-family:Space Grotesk,sans-serif;font-size:18px;font-weight:700;color:#FDF7F7">Waiting for Training Data</div>'
                    f'<div style="font-size:13.5px;color:#DFBFC3;margin-top:6px;max-width:680px;margin-left:auto;margin-right:auto;line-height:1.55">'
                    f'{html.escape(zone["summary"])} Diagnostics never shows guesswork epoch ranges before your model actually trains.'
                    f'</div>'
                    f'<div style="margin-top:10px;font-family:JetBrains Mono,monospace;font-size:12px;color:#C89EA4">{html.escape(zone["dataset_summary"])}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
        else:
            cand_badges = " ".join(
                f'<span class="nr-mono-badge" style="{"border-color:#E51923;color:#FDF7F7;background:#2A0D12;" if ep_c == zone["best_epoch"] else ""}">Epoch {ep_c}</span>'
                for ep_c in zone["candidate_epochs"]
            )
            best_loss_lbl = f"{zone['best_loss']:.3f}" if zone["best_loss"] is not None else "—"
            with st.container(border=True):
                st.markdown(
                    f'<div style="display:flex;flex-wrap:wrap;justify-content:space-between;align-items:center;gap:12px;margin-bottom:10px">'
                    f'<div><div style="font-size:12.5px;font-weight:600;color:#FF5A63;text-transform:uppercase;letter-spacing:0.04em">Recommended Best Epoch</div>'
                    f'<div style="font-family:Space Grotesk,sans-serif;font-size:28px;font-weight:700;color:#FDF7F7;line-height:1.15;margin-top:2px">Epoch {zone["best_epoch"]}</div>'
                    f'<div style="font-family:JetBrains Mono,monospace;font-size:12.5px;color:#DFBFC3;margin-top:4px">Checkpoint hint: {html.escape(zone["checkpoint_hint"])}</div></div>'
                    f'<div style="text-align:right"><div style="font-size:12px;color:#C89EA4;margin-bottom:5px">Best Checkpoints to Audition</div><div>{cand_badges}</div></div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

                zc1, zc2, zc3 = st.columns(3, gap="small")
                zc1.markdown(
                    f'<div class="nr-zone-card"><div style="font-size:12.5px;font-weight:600;color:#F59E0B">Best Mel Loss</div>'
                    f'<div style="font-family:JetBrains Mono,monospace;font-size:16px;font-weight:700;color:#FDF7F7;margin-top:4px">{best_loss_lbl}</div>'
                    f'<div style="font-size:12.5px;color:#C89EA4;margin-top:4px">{html.escape(zone["dataset_summary"])}</div></div>',
                    unsafe_allow_html=True,
                )
                zc2.markdown(
                    f'<div class="nr-zone-card nr-zone-sweet"><div style="font-size:12.5px;font-weight:600;color:#FDF7F7">Audition Window</div>'
                    f'<div style="font-family:JetBrains Mono,monospace;font-size:16px;font-weight:700;color:#FDF7F7;margin-top:4px">e{zone["sweet_min"]} – e{zone["sweet_max"]}</div>'
                    f'<div style="font-size:12.5px;color:#DFBFC3;margin-top:4px">Test saved checkpoints in Samples tab.</div></div>',
                    unsafe_allow_html=True,
                )
                zc3.markdown(
                    f'<div class="nr-zone-card"><div style="font-size:12.5px;font-weight:600;color:#FF2B35">Overtraining Watch</div>'
                    f'<div style="font-family:JetBrains Mono,monospace;font-size:16px;font-weight:700;color:#FDF7F7;margin-top:4px">{zone["over_label"]}</div>'
                    f'<div style="font-size:12.5px;color:#C89EA4;margin-top:4px">Watch for metallic timbre past this point.</div></div>',
                    unsafe_allow_html=True,
                )

            # Model Sound Feedback selector
            st.markdown('<div class="nr-field-title" style="margin:12px 0 6px 0">How does this checkpoint sound in Samples?</div>', unsafe_allow_html=True)
            sf_cols = st.columns(4, gap="small")
            sf_opts = [
                ("untested", "Not tested yet"),
                ("undertrained", "Undertrained (robotic / flat)"),
                ("clean", "Sounds clean & natural"),
                ("overtrained", "Overtrained (metallic / harsh)"),
            ]
            for idx_sf, (sf_id, sf_lbl) in enumerate(sf_opts):
                if sf_cols[idx_sf].button(sf_lbl, type="primary" if st.session_state["nr_diag_sound"] == sf_id else "secondary", use_container_width=True, key=f"nr_sf_{sf_id}"):
                    st.session_state["nr_diag_sound"] = sf_id
                    st.rerun()

            if parsed_diag and parsed_diag.get("live"):
                lp = parsed_diag["live"]
                st.markdown(
                    f'<div class="nr-prog-wrap">'
                    f'<div style="display:flex;justify-content:space-between;font-family:JetBrains Mono,monospace;font-size:12.5px">'
                    f'<span style="color:#FDF7F7;font-weight:600">Epoch {lp["epoch"]} · Step {lp["step"]} · Batch {lp["batch_done"]}/{lp["batch_total"]} ({lp["percent"]}%)</span>'
                    f'<span style="color:#DFBFC3">{lp["it_per_sec"]:.2f} it/s · {lp["elapsed"]} &lt; {lp["eta"]}</span></div>'
                    f'<div class="nr-prog-track"><div class="nr-prog-fill" style="width:{lp["percent"]}%"></div></div></div>',
                    unsafe_allow_html=True,
                )

            # Observed Mel Loss Chart (only rendered from real logged mel loss points)
            pts = [p for p in (parsed_diag["points"] if parsed_diag else []) if p.get("lowest_loss") is not None]
            if len(pts) >= 2:
                W, H, PAD_L, PAD_R, PAD_T, PAD_B = 1000, 180, 26, 26, 24, 26
                PLOT_W, PLOT_H = W - PAD_L - PAD_R, H - PAD_T - PAD_B
                min_ep = pts[0]["epoch"]
                max_ep = max(min_ep + sev_d, pts[-1]["epoch"])
                min_l = math.floor(min(p["lowest_loss"] for p in pts) * 0.92)
                max_l = math.ceil(max(p["lowest_loss"] for p in pts) * 1.05)
                span_l = max(0.5, max_l - min_l)
                to_x = lambda ep: PAD_L + ((min(max_ep, max(min_ep, ep)) - min_ep) / max(1, max_ep - min_ep)) * PLOT_W
                coords = [(to_x(p["epoch"]), PAD_T + (1.0 - (p["lowest_loss"] - min_l) / span_l) * PLOT_H, p["epoch"]) for p in pts]
                poly_pts = " ".join(f"{cx:.1f},{cy:.1f}" for cx, cy, _ in coords)
                sw_s = to_x(zone["sweet_min"])
                sw_e = to_x(zone["sweet_max"])
                best_x = to_x(zone["best_epoch"])
                dots_svg = "".join(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="3.5" fill="#FDF7F7" stroke="#E51923" stroke-width="2" />' for cx, cy, _ in coords)
                lbl_coords = [coords[0]] if coords else []
                for cx, cy, ep_v in coords[1:-1]:
                    if cx - lbl_coords[-1][0] >= 55 and coords[-1][0] - cx >= 55:
                        lbl_coords.append((cx, cy, ep_v))
                if len(coords) > 1:
                    lbl_coords.append(coords[-1])
                lbls_svg = "".join(f'<text x="{cx:.1f}" y="{H - 5}" text-anchor="middle" fill="#C89EA4" font-size="11" font-family="JetBrains Mono, monospace">e{ep_v}</text>' for cx, _, ep_v in lbl_coords)
                grid_svg = "".join(f'<line x1="{PAD_L}" y1="{PAD_T + PLOT_H * r:.1f}" x2="{W - PAD_R}" y2="{PAD_T + PLOT_H * r:.1f}" stroke="#47161D" stroke-dasharray="3 3" />' for r in (0.25, 0.5, 0.75))

                svg_html = (
                    f'<div style="background:#160709;border:1px solid #47161D;border-radius:8px;padding:12px;margin:10px 0">'
                    f'<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px">'
                    f'<span style="font-size:13px;font-weight:600;color:#FDF7F7">Observed Mel Loss &amp; Best Epoch (e{zone["best_epoch"]})</span>'
                    f'<span style="font-family:JetBrains Mono,monospace;font-size:11px;color:#C89EA4">● Best Epoch Window</span></div>'
                    f'<svg viewBox="0 0 {W} {H}" style="width:100%;height:auto">'
                    f'<rect x="{sw_s:.1f}" y="{PAD_T}" width="{max(8.0, sw_e - sw_s):.1f}" height="{PLOT_H}" fill="#E51923" fill-opacity="0.16" />'
                    f'{grid_svg}'
                    f'<line x1="{best_x:.1f}" y1="{PAD_T}" x2="{best_x:.1f}" y2="{PAD_T + PLOT_H}" stroke="#FF2B35" stroke-width="1.5" stroke-dasharray="4 3" />'
                    f'<text x="{best_x:.1f}" y="{PAD_T - 6}" text-anchor="middle" fill="#FDF7F7" font-size="11" font-family="JetBrains Mono, monospace">Best Epoch (e{zone["best_epoch"]})</text>'
                    f'<polyline fill="none" stroke="#E51923" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" points="{poly_pts}" />'
                    f'{dots_svg}{lbls_svg}</svg></div>'
                )
                st.markdown(svg_html, unsafe_allow_html=True)

            spd_note = f"Avg {parsed_diag['avg_speed_sec']}s/ep" if (parsed_diag and parsed_diag["avg_speed_sec"] > 0) else ""
            st.markdown(
                f'<div style="display:flex;justify-content:space-between;font-size:13.5px;color:#DFBFC3;margin-top:6px">'
                f'<span>{html.escape(zone["summary"])}</span>'
                f'<span style="font-family:JetBrains Mono,monospace;color:#C89EA4">{html.escape(spd_note)}</span></div>',
                unsafe_allow_html=True,
            )

        with st.expander("Optional: Import External .ipynb or Training Log", expanded=False):
            imp_file = st.file_uploader("Upload .ipynb / .log / .txt", type=["ipynb", "log", "txt"], key="nr_diag_up")
            if imp_file is not None:
                st.session_state["nr_diag_log_input"] = imp_file.getvalue().decode("utf-8", errors="replace")
                st.rerun()
            if st.session_state.get("nr_diag_log_input") and st.button("Reset to Live Training Log", type="secondary", key="nr_diag_clr"):
                st.session_state["nr_diag_log_input"] = ""
                st.rerun()

# ==============================================================================
# TAB 6: VOICE BLENDER (EXACT 3-STEP STUDIO PARITY)
# ==============================================================================
elif active_tab == "Voice Blender":
    _page_header("Voice Blender", "Merge two compatible RVC checkpoints into a single hybrid voice weights file.")
    pths_b = _list_pth_files()

    with _nr_card("nr_vb_step_1"):
        _step_head(1, "Source Models", "Load two .pth checkpoints via file drop or direct filesystem paths.")
        vb_l, vb_r = st.columns(2, gap="medium")
        with vb_l:
            up_a = st.file_uploader("Select primary .pth checkpoint (Model A)", type=["pth"], key="nr_vb_up_a")
            if up_a is not None:
                dst_a = Path(now_dir) / "logs" / up_a.name
                dst_a.parent.mkdir(parents=True, exist_ok=True)
                dst_a.write_bytes(up_a.getvalue())
                _clear_fs_caches()
                st.session_state["nr_vb_path_a"] = str(dst_a.relative_to(now_dir))
                st.session_state["nr_vb_in_a"] = str(dst_a.relative_to(now_dir))
            if pths_b:
                def _on_pick_vb_a() -> None:
                    pa_sel = st.session_state.get("nr_vb_pick_a", "")
                    if pa_sel:
                        st.session_state["nr_vb_path_a"] = pa_sel
                        st.session_state["nr_vb_in_a"] = pa_sel
                        st.session_state["nr_vb_pick_a"] = ""

                st.selectbox("Library Model A", options=[""] + pths_b, format_func=lambda x: "Pick from logs/..." if x == "" else x, on_change=_on_pick_vb_a, key="nr_vb_pick_a")
            st.markdown('<div class="nr-field-title">Path to first model</div><div class="nr-field-help">Or type a relative logs/ path directly.</div>', unsafe_allow_html=True)
            st.session_state["nr_vb_path_a"] = st.text_input("Path to first model", value=st.session_state.get("nr_vb_path_a", ""), placeholder="Enter path to model", label_visibility="collapsed", key="nr_vb_in_a")
        with vb_r:
            up_b = st.file_uploader("Select secondary .pth checkpoint (Model B)", type=["pth"], key="nr_vb_up_b")
            if up_b is not None:
                dst_b = Path(now_dir) / "logs" / up_b.name
                dst_b.parent.mkdir(parents=True, exist_ok=True)
                dst_b.write_bytes(up_b.getvalue())
                _clear_fs_caches()
                st.session_state["nr_vb_path_b"] = str(dst_b.relative_to(now_dir))
                st.session_state["nr_vb_in_b"] = str(dst_b.relative_to(now_dir))
            if pths_b:
                def _on_pick_vb_b() -> None:
                    pb_sel = st.session_state.get("nr_vb_pick_b", "")
                    if pb_sel:
                        st.session_state["nr_vb_path_b"] = pb_sel
                        st.session_state["nr_vb_in_b"] = pb_sel
                        st.session_state["nr_vb_pick_b"] = ""

                st.selectbox("Library Model B", options=[""] + pths_b, format_func=lambda x: "Pick from logs/..." if x == "" else x, on_change=_on_pick_vb_b, key="nr_vb_pick_b")
            st.markdown('<div class="nr-field-title">Path to second model</div><div class="nr-field-help">Or type a relative logs/ path directly.</div>', unsafe_allow_html=True)
            st.session_state["nr_vb_path_b"] = st.text_input("Path to second model", value=st.session_state.get("nr_vb_path_b", ""), placeholder="Enter path to model", label_visibility="collapsed", key="nr_vb_in_b")

    with _nr_card("nr_vb_step_2"):
        _step_head(2, "Blend Settings", "Define the merged checkpoint filename and weight mixing balance.")
        bs_l, bs_r = st.columns(2, gap="medium")
        with bs_l:
            st.markdown('<div class="nr-field-title">Model Name</div><div class="nr-field-help">Identifier assigned to the merged .pth export.</div>', unsafe_allow_html=True)
            blend_name = st.text_input("Model Name", value=st.session_state.get("nr_vb_name", ""), placeholder="Enter model name", label_visibility="collapsed", key="nr_vb_name")
        with bs_r:
            st.markdown('<div class="nr-field-title">Sampling Rate</div><div class="nr-field-help">Must match the target rate of both source checkpoints.</div>', unsafe_allow_html=True)
            blend_sr = st.selectbox("Sampling Rate", ["32000", "40000", "48000"], label_visibility="collapsed", key="nr_vb_sr")
        with st.container(border=True):
            br_l, br_val, br_rst = st.columns([4.4, 0.85, 0.55], gap="small")
            br_l.markdown('<div class="nr-field-title">Blend Ratio</div><div class="nr-field-help">Balance influence between checkpoints: 0 leans 100% toward Model B, while 1 leans 100% toward Model A.</div>', unsafe_allow_html=True)
            br_val.markdown(f'<div style="background:#160709;border:1px solid #47161D;border-radius:8px;padding:6px 10px;text-align:center;font-family:JetBrains Mono,monospace;font-size:13px;color:#FDF7F7">{float(st.session_state.get("nr_vb_ratio", 0.5)):g}</div>', unsafe_allow_html=True)
            if br_rst.button("↺", key="nr_vb_rst_ratio", help="Reset Blend Ratio to 0.5", use_container_width=True):
                st.session_state["nr_vb_ratio"] = 0.5
                st.rerun()
            blend_ratio = st.slider("Blend Ratio", 0.0, 1.0, float(st.session_state.get("nr_vb_ratio", 0.5)), step=0.05, label_visibility="collapsed", key="nr_vb_ratio_sl")
            st.session_state["nr_vb_ratio"] = blend_ratio

    with _nr_card("nr_vb_step_3"):
        _step_head(3, "Blend & Export", "Execute the blend and download the packaged recipe or shell command.")
        if st.button(_tr("Fusion"), type="primary", use_container_width=True, key="nr_btn_fusion"):
            pa = st.session_state.get("nr_vb_path_a", "").strip()
            pb = st.session_state.get("nr_vb_path_b", "").strip()
            bn = re.sub(r"[^a-zA-Z0-9_-]+", "-", blend_name.strip()) or "BlendedVoice"
            if not pa or not pb:
                st.warning("Provide both Model A and Model B .pth checkpoints.")
            else:
                with st.spinner("Blending model weights..."):
                    rc, msg = _run_core_sync(
                        "run_model_blender_script",
                        {
                            "model_name": bn,
                            "pth_path_1": pa if pa.startswith("/") else os.path.join(now_dir, pa),
                            "pth_path_2": pb if pb.startswith("/") else os.path.join(now_dir, pb),
                            "ratio": float(blend_ratio),
                        },
                    )
                    st.session_state["nr_blend_log"] = msg
                    cand = Path(now_dir) / "logs" / f"{bn}.pth"
                    if cand.is_file():
                        st.session_state["nr_blend_out"] = str(cand)
                st.rerun()

        _output_box("Blend output", st.session_state["nr_blend_log"])
        vb_b1, _, vb_b2 = st.columns([1.4, 2.8, 1.6], gap="small")
        with vb_b1:
            if st.session_state["nr_blend_out"] and os.path.isfile(st.session_state["nr_blend_out"]):
                bp = Path(st.session_state["nr_blend_out"])
                st.download_button(_tr("Download Model"), data=bp.read_bytes(), file_name=bp.name, type="secondary", use_container_width=True, key="nr_dl_blend")
            else:
                st.button(_tr("Download Model"), disabled=True, type="secondary", use_container_width=True, key="nr_dl_blend_dis")
        with vb_b2:
            with st.expander("View Blend CLI", expanded=False):
                st.code(
                    f'python core.py model_blend --model_name "{blend_name.strip() or "BlendedVoice"}" --pth_path_1 "{st.session_state.get("nr_vb_path_a", "logs/ModelA.pth")}" --pth_path_2 "{st.session_state.get("nr_vb_path_b", "logs/ModelB.pth")}" --ratio {blend_ratio:.2f}',
                    language="bash",
                )

# ==============================================================================
# TAB 7: TTS (EXACT SINGLE-CARD STUDIO PARITY)
# ==============================================================================
elif active_tab == "TTS":
    _page_header("TTS", "Synthesize spoken dialogue with Edge-TTS and route it through your selected RVC voice checkpoint.", show_readme=False)
    pths_t = _list_pth_files()
    idxs_t = _list_index_files()
    voices_t = _load_tts_voices()

    with _nr_card("nr_tts_step_1"):
        _step_head(1, "Voice & Script", "Choose an Edge-TTS narrator, pair your trained RVC voice model, and write the spoken script.")
        tc1, tc2 = st.columns(2, gap="medium")
        with tc1:
            st.markdown('<div class="nr-field-title">Base Edge-TTS Speaker</div><div class="nr-field-help">Neural speaker voice before converting through your RVC model.</div>', unsafe_allow_html=True)
            tts_voice = st.selectbox("Base Edge-TTS Speaker", options=voices_t, index=0, label_visibility="collapsed", key="nr_tts_voice_sel")
        with tc2:
            st.markdown('<div class="nr-field-title">Target RVC Voice Model (Synced with Models)</div><div class="nr-field-help">Optional trained .pth checkpoint to apply your custom voice timbre.</div>', unsafe_allow_html=True)
            tts_pth = st.selectbox("Target RVC Voice Model (Synced with Models)", options=[""] + pths_t, format_func=lambda x: "-- Select Target .pth Model --" if x == "" else x, label_visibility="collapsed", key="nr_tts_pth_sel")

        st.markdown('<div class="nr-field-title">Index File (Optional)</div><div class="nr-field-help">FAISS retrieval index to strengthen vocal accent and tone similarity.</div>', unsafe_allow_html=True)
        tts_idx = st.selectbox("Index File (Optional)", options=[""] + idxs_t, format_func=lambda x: "-- Optional .index File --" if x == "" else x, label_visibility="collapsed", key="nr_tts_idx_sel")
        st.markdown('<div class="nr-field-title">Text to Speak</div><div class="nr-field-help">Dialogue or test lines rendered into speech and routed through RVC.</div>', unsafe_allow_html=True)
        tts_text = st.text_area("Text to Speak", value="Welcome to NoRegretsRVC. Testing synthesized speech with your voice model.", height=96, label_visibility="collapsed", key="nr_tts_text_in")

        ts1, ts2 = st.columns(2, gap="medium")
        with ts1:
            with st.container(border=True):
                st.markdown('<div class="nr-field-title">Pitch Shift (Semitones)</div><div class="nr-field-help">Adjust target octave/register up or down (-12 to +12).</div>', unsafe_allow_html=True)
                tts_pitch = st.slider("Pitch Shift (Semitones)", -12, 12, 0, step=1, label_visibility="collapsed", key="nr_tts_pitch")
            with st.container(border=True):
                st.markdown('<div class="nr-field-title">Index Feature Ratio</div><div class="nr-field-help">Accent strength pulled from your trained .index database.</div>', unsafe_allow_html=True)
                tts_idx_r = st.slider("Index Feature Ratio", 0.0, 1.0, 0.75, step=0.05, label_visibility="collapsed", key="nr_tts_idx_r")
        with ts2:
            with st.container(border=True):
                st.markdown('<div class="nr-field-title">Speech Speed (%)</div><div class="nr-field-help">Pace of the base Edge-TTS delivery before voice conversion.</div>', unsafe_allow_html=True)
                tts_rate = st.slider("Speech Speed (%)", -50, 50, 0, step=5, label_visibility="collapsed", key="nr_tts_rate")
            with st.container(border=True):
                st.markdown('<div class="nr-field-title">Consonant Protect</div><div class="nr-field-help">Preserves crisp breaths and sharp consonant articulation.</div>', unsafe_allow_html=True)
                tts_prot = st.slider("Consonant Protect", 0.0, 0.5, 0.5, step=0.01, label_visibility="collapsed", key="nr_tts_prot")

        if st.button(_tr("Convert Text-to-Speech"), type="primary", use_container_width=True, key="nr_btn_run_tts"):
            if not tts_text.strip():
                st.warning("Enter text to speak first.")
            else:
                out_tts = os.path.join(now_dir, "assets", "audios", "tts_output.wav")
                out_rvc = os.path.join(now_dir, "assets", "audios", "tts_rvc_output.wav")
                with st.spinner("Synthesizing Edge-TTS & converting voice..."):
                    rc, msg = _run_core_sync(
                        "run_tts_script",
                        {
                            "tts_file": "",
                            "tts_text": tts_text.strip(),
                            "tts_voice": tts_voice,
                            "tts_rate": int(tts_rate),
                            "pitch": int(tts_pitch),
                            "index_rate": float(tts_idx_r),
                            "volume_envelope": 1.0,
                            "protect": float(tts_prot),
                            "f0_method": "rmvpe",
                            "output_tts_path": out_tts,
                            "output_rvc_path": out_rvc,
                            "pth_path": os.path.join(now_dir, tts_pth) if tts_pth else "",
                            "index_path": os.path.join(now_dir, tts_idx) if tts_idx else "",
                            "f0_autotune": False,
                            "f0_autotune_strength": 1.0,
                            "proposed_pitch": False,
                            "proposed_pitch_threshold": 155.0,
                            "export_format": "WAV",
                            "embedder_model": "contentvec",
                        },
                    )
                    st.session_state["nr_tts_log"] = msg
                    target_au = out_rvc if (tts_pth and os.path.isfile(out_rvc)) else out_tts
                    if rc == 0 and os.path.isfile(target_au):
                        st.session_state["nr_tts_out"] = target_au
                st.rerun()

        if st.session_state["nr_tts_log"]:
            _output_box("TTS Output", st.session_state["nr_tts_log"])
        if st.session_state["nr_tts_out"] and os.path.isfile(st.session_state["nr_tts_out"]):
            tp = Path(st.session_state["nr_tts_out"])
            st.audio(str(tp))
            st.download_button(f"Download {tp.name}", data=tp.read_bytes(), file_name=tp.name, type="secondary", key="nr_dl_tts")

# ==============================================================================
# TAB 8: SETTINGS (GENERAL & TRAINING — INSTANT AUTO-SAVE LANGUAGE & FP32 DEFAULT)
# ==============================================================================
elif active_tab == "Settings":
    sh_l, sh_r = st.columns([4, 1.5], gap="small")
    with sh_l:
        _page_header("Settings", "Customize engine preferences, training precision, and Kaggle tunnel defaults.", show_readme=False)
    with sh_r:
        sg_b, st_b = st.columns(2, gap="small")
        if sg_b.button(_tr("General"), type="primary" if st.session_state["nr_settings_subtab"] == "General" else "secondary", use_container_width=True, key="nr_set_sub_gen"):
            st.session_state["nr_settings_subtab"] = "General"
            st.rerun()
        if st_b.button(_tr("Training"), type="primary" if st.session_state["nr_settings_subtab"] == "Training" else "secondary", use_container_width=True, key="nr_set_sub_trn"):
            st.session_state["nr_settings_subtab"] = "Training"
            st.rerun()

    cfg_cur = _load_config()

    if st.session_state["nr_settings_subtab"] == "General":
        with _nr_card("nr_set_gen_1"):
            _step_head(1, "List Filtering", "Display a search bar to quickly filter long checkpoint and index dropdowns.")
            with st.container(border=True):
                lf_l, lf_r = st.columns([5.2, 0.8], gap="small")
                lf_l.markdown(
                    '<div class="nr-field-title">Enable model/index list filter</div>'
                    '<div class="nr-field-help" style="margin-bottom:0">Automatically off by default to keep model and index selectors clean without extra search boxes.</div>',
                    unsafe_allow_html=True,
                )
                lf_val = lf_r.checkbox("Enable model/index list filter", value=bool(st.session_state["nr_list_filter"]), label_visibility="collapsed", key="nr_set_lf_chk")
                if lf_val != bool(st.session_state["nr_list_filter"]):
                    st.session_state["nr_list_filter"] = lf_val
                    cfg_cur["filter"] = bool(lf_val)
                    _save_config(cfg_cur)
                    st.rerun()

        with _nr_card("nr_set_gen_2"):
            _step_head(2, "Pitch Correction", "Prevent high-note octave flips during RMVPE F0 pitch extraction.")
            with st.expander("RMVPE high-register correction", expanded=False):
                rmv_on = st.checkbox("Enable RMVPE high-register correction", value=bool(st.session_state["nr_rmvpe_corr"]), key="nr_set_rmv_on")
                rc1, rc2 = st.columns(2, gap="medium")
                rmv_mode = rc1.selectbox("Correction mode", ["true_pitch", "fold"], index=0 if st.session_state["nr_rmvpe_mode"] == "true_pitch" else 1, key="nr_set_rmv_mode")
                rmv_ceil = rc2.slider("F0 ceiling (Hz)", 1000, 2000, int(st.session_state["nr_rmvpe_ceil"]), step=25, key="nr_set_rmv_ceil")
                if rmv_on != st.session_state["nr_rmvpe_corr"] or rmv_mode != st.session_state["nr_rmvpe_mode"] or rmv_ceil != st.session_state["nr_rmvpe_ceil"]:
                    st.session_state["nr_rmvpe_corr"] = rmv_on
                    st.session_state["nr_rmvpe_mode"] = rmv_mode
                    st.session_state["nr_rmvpe_ceil"] = rmv_ceil
                    cfg_cur["rmvpe_high_register"] = {"enabled": bool(rmv_on), "mode": rmv_mode, "ceiling_hz": int(rmv_ceil)}
                    _save_config(cfg_cur)

        with _nr_card("nr_set_gen_3"):
            _step_head(3, "Discord Integration", "Broadcast your active studio session to Discord Rich Presence.")
            with st.container(border=True):
                dc_l, dc_r = st.columns([5.2, 0.8], gap="small")
                dc_l.markdown(
                    '<div class="nr-field-title">Enable Discord Rich Presence</div>'
                    '<div class="nr-field-help" style="margin-bottom:0">Displays your active training or sampling status on your Discord profile.</div>',
                    unsafe_allow_html=True,
                )
                dc_val = dc_r.checkbox("Enable Discord Rich Presence", value=bool(st.session_state["nr_discord"]), label_visibility="collapsed", key="nr_set_dc_chk")
                if dc_val != bool(st.session_state["nr_discord"]):
                    st.session_state["nr_discord"] = dc_val
                    cfg_cur["discord_presence"] = bool(dc_val)
                    cfg_cur["nr_discord_initialized"] = True
                    _save_config(cfg_cur)

        with _nr_card("nr_set_gen_4"):
            _step_head(4, "Language", "Switches the studio interface instantaneously and sets your exported WebUI locale.")
            lang_codes = [c for c, _ in SUPPORTED_LANGUAGES]
            cur_code = st.session_state.get("nr_lang", "System Auto")

            def _on_lang_change() -> None:
                chosen = st.session_state["nr_set_lang_sel"]
                st.session_state["nr_lang"] = chosen
                cfg_l = _load_config()
                cfg_l["lang"] = {"override": chosen != "System Auto", "selected_lang": chosen}
                _save_config(cfg_l)

            st.selectbox(
                _tr("Language"),
                options=lang_codes,
                index=lang_codes.index(cur_code) if cur_code in lang_codes else 0,
                format_func=lambda c: LANG_LABELS.get(c, c),
                on_change=_on_lang_change,
                key="nr_set_lang_sel",
            )

        with _nr_card("nr_set_gen_5"):
            _step_head(5, "Updates & System", "Check the newest NoRegretsRVC version or trigger a Kaggle WebUI restart.")
            st.markdown(
                f'<div class="nr-field-title">Version Checker</div>'
                f'<div class="nr-field-help">Verifies your installed NoRegretsRVC build against the newest available release (currently {html.escape(APP_VERSION)}).</div>',
                unsafe_allow_html=True,
            )
            st.markdown(f'<div class="nr-output-box" style="margin-bottom:10px">{html.escape(st.session_state["nr_ver_status"])}</div>', unsafe_allow_html=True)
            u1, u2, _ = st.columns([1.45, 1.35, 3.2], gap="small")
            if u1.button(_tr("Check for updates"), type="secondary", use_container_width=True, key="nr_set_chk_upd"):
                try:
                    req = urllib.request.Request("https://api.github.com/repos/HunterThatsMe/NoRegretsRVC/commits/main", headers={"User-Agent": "NoRegretsRVC"})
                    with urllib.request.urlopen(req, timeout=8) as resp:
                        data = json.loads(resp.read().decode("utf-8"))
                        sha = str(data.get("sha", ""))[:7]
                        st.session_state["nr_ver_status"] = f"NoRegretsRVC Beta version {APP_VERSION} · Latest GitHub commit: {sha}."
                except Exception as exc:
                    st.session_state["nr_ver_status"] = f"NoRegretsRVC Beta version {APP_VERSION} (offline check: {exc})."
                st.rerun()
            if u2.button(_tr("Restart Applio"), type="primary", use_container_width=True, key="nr_set_restart"):
                st.session_state["nr_ver_status"] = "Restarting WebUI session state cleanly..."
                st.rerun()
    else:
        with _nr_card("nr_set_trn_1"):
            _step_head(1, "Model Author", "Credit written into exported model metadata headers.")
            st.markdown('<div class="nr-field-title">Model Author Name</div><div class="nr-field-help">Creator handle embedded in published checkpoints.</div>', unsafe_allow_html=True)
            st.session_state["nr_author"] = st.text_input("Model Author Name", value=st.session_state["nr_author"], label_visibility="collapsed", key="nr_set_author_inp")
            _output_box("Model author output", st.session_state["nr_author_out"])
            if st.button(_tr("Set name"), type="primary", key="nr_set_author_btn"):
                cleaned_a = st.session_state["nr_author"].strip() or "None"
                st.session_state["nr_author"] = cleaned_a
                cfg_cur["model_author"] = cleaned_a
                _save_config(cfg_cur)
                st.session_state["nr_author_out"] = f'Model author set to "{cleaned_a}".'
                st.rerun()

        with _nr_card("nr_set_trn_2"):
            _step_head(2, "Training Precision", "Control floating-point accuracy across training and voice conversion.")
            st.markdown('<div class="nr-field-title">Precision</div><div class="nr-field-help">FP32 prioritizes maximum stability on T4 GPUs; FP16 lowers VRAM usage.</div>', unsafe_allow_html=True)
            prec_opts = ["fp32", "fp16"]
            cur_p = str(st.session_state.get("nr_precision", "fp32"))
            st.session_state["nr_precision"] = st.selectbox("Precision", options=prec_opts, index=prec_opts.index(cur_p) if cur_p in prec_opts else 0, label_visibility="collapsed", key="nr_set_prec_sel")
            _output_box("Precision output", st.session_state["nr_precision_out"])
            if st.button(_tr("Update precision"), type="primary", key="nr_set_prec_btn"):
                chosen_p = st.session_state["nr_precision"]
                cfg_cur["precision"] = chosen_p
                cfg_cur["nr_precision_initialized"] = True
                _save_config(cfg_cur)
                st.session_state["nr_precision_out"] = f"Training & inference precision updated to {chosen_p.upper()}."
                st.rerun()

        with _nr_card("nr_set_trn_3"):
            _step_head(3, "Kaggle Tunnel & Runtime Hardening", "Active Streamlit WebUI runtime, precision, and CUDA session details.")
            rt1, rt2 = st.columns(2, gap="small")
            rt1.markdown(f'<div class="nr-metric-box"><div class="nr-metric-k">Working Directory</div><div class="nr-metric-v">{html.escape(now_dir)}</div></div>', unsafe_allow_html=True)
            rt2.markdown(f'<div class="nr-metric-box"><div class="nr-metric-k">Active Precision</div><div class="nr-metric-v">{html.escape(st.session_state["nr_precision"].upper())}</div></div>', unsafe_allow_html=True)
            rt3, rt4 = st.columns(2, gap="small")
            rt3.markdown(f'<div class="nr-metric-box"><div class="nr-metric-k">GPU Allocation</div><div class="nr-metric-v">{html.escape(_gpu_info_lines(st.session_state["nr_gpu_num"]).replace(chr(10), " · "))}</div></div>', unsafe_allow_html=True)
            rt4.markdown(f'<div class="nr-metric-box"><div class="nr-metric-k">Python Environment</div><div class="nr-metric-v">{html.escape(sys.executable)}</div></div>', unsafe_allow_html=True)

_tab_stack.close()
`;
}
