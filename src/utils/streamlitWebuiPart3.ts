export function buildStreamlitTabsPart3(): string {
  return `
# ==============================================================================
# TAB 3: SAMPLES (4-STEP WORKFLOW: VOICE, SOURCE, PITCH & SETTINGS, CONVERT)
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

    _s1_pth = str(st.session_state.get("nr_inf_pth", "")).strip()
    _s2_mode = str(st.session_state.get("nr_inf_mode", "Audio Files"))
    _s2_q = st.session_state.get("nr_sample_queue", [])
    _s2_ready = bool(_s2_q) if _s2_mode == "Audio Files" else bool(str(st.session_state.get("nr_batch_folder", "")).strip())
    _s3_p = int(st.session_state.get("nr_def_pitch", 0))
    _s3_p_txt = f"+{_s3_p}" if _s3_p > 0 else str(_s3_p)
    _s3_idx = float(st.session_state.get("nr_def_idx_rate", 0.75))
    _s4_n = len(st.session_state.get("nr_saved_samples", []))
    _rc1, _rc2, _rc3, _rc4 = st.columns(4, gap="small")
    _rc1.markdown(
        f'<div class="nr-metric-box" style="{"border-color:rgba(16,185,129,0.55);background:#081A14;" if _s1_pth else ""}">'
        f'<div class="nr-metric-k">1. Voice Model</div>'
        f'<div class="nr-metric-v">{html.escape(Path(_s1_pth).name) if _s1_pth else "Pick .pth below"}</div></div>',
        unsafe_allow_html=True,
    )
    _rc2.markdown(
        f'<div class="nr-metric-box" style="{"border-color:rgba(16,185,129,0.55);background:#081A14;" if _s2_ready else ""}">'
        f'<div class="nr-metric-k">2. Audio Source</div>'
        f'<div class="nr-metric-v">{f"{len(_s2_q)} clip(s) queued" if _s2_mode == "Audio Files" and _s2_q else ("Batch folder ready" if _s2_mode == "Batch" and _s2_ready else "Upload or quick-pick")}</div></div>',
        unsafe_allow_html=True,
    )
    _rc3.markdown(
        f'<div class="nr-metric-box"><div class="nr-metric-k">3. Pitch &amp; Similarity</div>'
        f'<div class="nr-metric-v">Pitch {_s3_p_txt} · Index {_s3_idx:.2f}</div></div>',
        unsafe_allow_html=True,
    )
    _rc4.markdown(
        f'<div class="nr-metric-box" style="{"border-color:rgba(16,185,129,0.55);background:#081A14;" if _s4_n > 0 else ""}">'
        f'<div class="nr-metric-k">4. Convert &amp; Output</div>'
        f'<div class="nr-metric-v">{f"{_s4_n} sample(s) ready" if _s4_n > 0 else ("Ready to convert" if _s1_pth and _s2_ready else "Complete Steps 1–2")}</div></div>',
        unsafe_allow_html=True,
    )

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

    # --- STEP 2 & STEP 3 SHARED WORKSPACE FRAGMENT (INSTANT BIDIRECTIONAL PITCH SYNC) ---
    def _audio_mime(p_obj: Path) -> str:
        ext = p_obj.suffix.lower().lstrip(".")
        if ext == "mp3":
            return "audio/mpeg"
        if ext == "m4a":
            return "audio/mp4"
        return f"audio/{ext or 'wav'}"

    def _sync_samples_state() -> None:
        adv_map = [
            ("nr_adv_idx", "nr_def_idx_rate", float, 0.75),
            ("nr_adv_vol", "nr_def_vol_env", float, 1.0),
            ("nr_adv_prot", "nr_def_protect", float, 0.5),
            ("nr_adv_sid", "nr_def_sid", int, 0),
            ("nr_adv_f0", "nr_def_f0", str, "rmvpe"),
            ("nr_adv_emb", "nr_def_emb", str, "contentvec"),
            ("nr_adv_at", "nr_def_autotune", bool, False),
            ("nr_adv_pp", "nr_def_proposed", bool, False),
            ("nr_adv_sp", "nr_def_split", bool, False),
            ("nr_adv_clean", "nr_def_clean", bool, False),
            ("nr_adv_cleandb", "nr_def_clean_db", int, 12),
            ("nr_adv_fmt_on", "nr_def_formant", bool, False),
            ("nr_adv_qf", "nr_def_qf", float, 1.0),
            ("nr_adv_tf", "nr_def_tf", float, 1.0),
            ("nr_adv_outp", "nr_def_out_path", str, "assets/audios/output.wav"),
            ("nr_adv_fmt", "nr_def_fmt", str, "WAV"),
        ]
        for w_key, state_key, caster, fallback_v in adv_map:
            if state_key not in st.session_state:
                st.session_state[state_key] = fallback_v
            if w_key in st.session_state:
                st.session_state[state_key] = caster(st.session_state[w_key])
            else:
                st.session_state[w_key] = caster(st.session_state[state_key])

        q_list = st.session_state.get("nr_sample_queue", [])
        for c in q_list:
            cid = c["id"]
            cp_k = f"nr_cp_{cid}"
            ci_k = f"nr_ci_{cid}"
            cprot_k = f"nr_cprot_{cid}"
            cvol_k = f"nr_cvol_{cid}"
            cat_k = f"nr_cat_{cid}"
            cpp_k = f"nr_cpp_{cid}"
            csp_k = f"nr_csp_{cid}"
            cf_k = f"nr_cf_{cid}"
            cm_k = f"nr_cm_{cid}"
            cx_k = f"nr_cx_{cid}"
            if cp_k in st.session_state:
                nv = int(st.session_state[cp_k])
                if nv != int(c.get("pitch", 0)):
                    c["pitch"] = nv
                    c["custom_pitch"] = True
            if ci_k in st.session_state:
                nv_f = float(st.session_state[ci_k])
                if abs(nv_f - float(c.get("index_rate", 0.75))) > 1e-6:
                    c["index_rate"] = nv_f
                    c["custom_index"] = True
            if cprot_k in st.session_state:
                nv_f = float(st.session_state[cprot_k])
                if abs(nv_f - float(c.get("protect", 0.5))) > 1e-6:
                    c["protect"] = nv_f
                    c["custom_protect"] = True
            if cvol_k in st.session_state:
                nv_f = float(st.session_state[cvol_k])
                if abs(nv_f - float(c.get("volume_envelope", 1.0))) > 1e-6:
                    c["volume_envelope"] = nv_f
                    c["custom_vol"] = True
            if cat_k in st.session_state:
                nv_b = bool(st.session_state[cat_k])
                if nv_b != bool(c.get("autotune", False)):
                    c["autotune"] = nv_b
                    c["custom_at"] = True
            if cpp_k in st.session_state:
                nv_b = bool(st.session_state[cpp_k])
                if nv_b != bool(c.get("proposed_pitch", False)):
                    c["proposed_pitch"] = nv_b
                    c["custom_pp"] = True
            if csp_k in st.session_state:
                nv_b = bool(st.session_state[csp_k])
                if nv_b != bool(c.get("split_audio", False)):
                    c["split_audio"] = nv_b
                    c["custom_sp"] = True
            if cf_k in st.session_state:
                nv_s = str(st.session_state[cf_k]).upper()
                if nv_s != str(c.get("format", "WAV")).upper():
                    c["format"] = nv_s
                    c["custom_fmt"] = True
            if cm_k in st.session_state:
                c["model"] = str(st.session_state[cm_k])
            if cx_k in st.session_state:
                c["index"] = str(st.session_state[cx_k])

        def_specs = [
            ("nr_def_pitch", "pitch", "nr_cp_", "custom_pitch", int, 0),
            ("nr_def_idx_rate", "index_rate", "nr_ci_", "custom_index", float, 0.75),
            ("nr_def_protect", "protect", "nr_cprot_", "custom_protect", float, 0.5),
            ("nr_def_vol_env", "volume_envelope", "nr_cvol_", "custom_vol", float, 1.0),
            ("nr_def_autotune", "autotune", "nr_cat_", "custom_at", bool, False),
            ("nr_def_proposed", "proposed_pitch", "nr_cpp_", "custom_pp", bool, False),
            ("nr_def_split", "split_audio", "nr_csp_", "custom_sp", bool, False),
            ("nr_def_fmt", "format", "nr_cf_", "custom_fmt", str, "WAV"),
        ]
        single_clip = len(q_list) == 1
        for def_key, clip_field, w_prefix, custom_flag, caster, fallback_v in def_specs:
            cur_v = caster(st.session_state.get(def_key, fallback_v))
            if caster is str:
                cur_v = cur_v.upper()
            prev_key = f"_nr_prev_{def_key}"
            if prev_key not in st.session_state:
                st.session_state[prev_key] = cur_v
                continue
            prev_v = caster(st.session_state[prev_key])
            if caster is str:
                prev_v = prev_v.upper()
            changed = (abs(cur_v - prev_v) > 1e-6) if caster is float else (cur_v != prev_v)
            if changed:
                st.session_state[prev_key] = cur_v
                for c in q_list:
                    if single_clip or not bool(c.get(custom_flag, False)):
                        c[clip_field] = cur_v
                        st.session_state[f"{w_prefix}{c['id']}"] = cur_v
                        c[custom_flag] = False

    def _clear_clip_widget_keys(cid: str) -> None:
        for pfx in ("nr_cp_", "nr_ci_", "nr_cprot_", "nr_cvol_", "nr_cat_", "nr_cpp_", "nr_csp_", "nr_cf_", "nr_cm_", "nr_cx_"):
            st.session_state.pop(f"{pfx}{cid}", None)

    def _render_samples_workspace_body() -> None:
        _sync_samples_state()

        # --- STEP 2: Audio Source & Multi-Sample Queue ---
        with _nr_card("nr_samples_step_2"):
            s2_l, s2_r = st.columns([3.5, 1.5], gap="small")
            with s2_l:
                _step_head(2, "Audio Source", "Upload or add recordings below, or switch to Batch to convert a whole folder. Pitch & settings live in Step 3.")
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
                seen_ups = st.session_state.setdefault("nr_inf_seen_ups", set())
                active_sigs = {f"{uf.name}:{uf.size}" for uf in (up_clips or [])}
                seen_ups.intersection_update(active_sigs)
                if up_clips:
                    au_dir = Path(now_dir) / "assets" / "audios"
                    au_dir.mkdir(parents=True, exist_ok=True)
                    queued_paths = {c["path"] for c in st.session_state["nr_sample_queue"]}
                    added_n = 0
                    for uf in up_clips:
                        up_sig = f"{uf.name}:{uf.size}"
                        if up_sig in seen_ups:
                            continue
                        seen_ups.add(up_sig)
                        dst = au_dir / uf.name
                        if not dst.is_file():
                            dst.write_bytes(uf.getvalue())
                        rel_dst = str(dst.relative_to(now_dir))
                        if rel_dst not in queued_paths:
                            st.session_state["nr_sample_queue"].append({
                                "id": _next_clip_id("clip"),
                                "name": uf.name,
                                "path": rel_dst,
                                "pitch": int(st.session_state.get("nr_def_pitch", 0)),
                                "index_rate": float(st.session_state.get("nr_def_idx_rate", 0.75)),
                                "protect": float(st.session_state.get("nr_def_protect", 0.5)),
                                "volume_envelope": float(st.session_state.get("nr_def_vol_env", 1.0)),
                                "autotune": bool(st.session_state.get("nr_def_autotune", False)),
                                "proposed_pitch": bool(st.session_state.get("nr_def_proposed", False)),
                                "split_audio": bool(st.session_state.get("nr_def_split", False)),
                                "format": str(st.session_state.get("nr_def_fmt", "WAV")).upper(),
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
                                "id": _next_clip_id("clip"),
                                "name": Path(pick_au).name,
                                "path": pick_au,
                                "pitch": int(st.session_state.get("nr_def_pitch", 0)),
                                "index_rate": float(st.session_state.get("nr_def_idx_rate", 0.75)),
                                "protect": float(st.session_state.get("nr_def_protect", 0.5)),
                                "volume_envelope": float(st.session_state.get("nr_def_vol_env", 1.0)),
                                "autotune": bool(st.session_state.get("nr_def_autotune", False)),
                                "proposed_pitch": bool(st.session_state.get("nr_def_proposed", False)),
                                "split_audio": bool(st.session_state.get("nr_def_split", False)),
                                "format": str(st.session_state.get("nr_def_fmt", "WAV")).upper(),
                                "model": "",
                                "index": "",
                                "converted_path": "",
                            })
                            st.rerun()

                # Inline Quick Test Voice (Free Samples) so users never have to leave Samples tab
                st.markdown('<div class="nr-field-title" style="margin-top:6px">Quick Test Voice (Free Samples — No Upload Needed)</div>', unsafe_allow_html=True)
                fs_quick_cols = st.columns(4, gap="small")
                for _fq_idx, _fq_item in enumerate(FREE_SAMPLES_CATALOG[:4]):
                    _short_lbl = _fq_item["title"].replace(" Speaking ", " ")
                    if fs_quick_cols[_fq_idx].button(f"+ {_short_lbl}", type="secondary", use_container_width=True, key=f"nr_quick_fs_{_fq_item['id']}"):
                        _fq_dir = Path(now_dir) / "assets" / "free_samples"
                        _fq_dir.mkdir(parents=True, exist_ok=True)
                        _fq_wav = _fq_dir / _fq_item["repoWav"]
                        if not _fq_wav.is_file():
                            try:
                                urllib.request.urlretrieve(f"https://raw.githubusercontent.com/HunterThatsMe/NoRegretsRVC/main/public/assets/aistudio/{_fq_item['repoWav']}", _fq_wav)
                            except Exception:
                                pass
                        st.session_state["nr_sample_queue"].append({
                            "id": _next_clip_id(f"qfs-{_fq_item['id']}"),
                            "name": _fq_item["fileName"],
                            "path": str(_fq_wav.relative_to(now_dir)),
                            "pitch": int(st.session_state.get("nr_def_pitch", 0)),
                            "index_rate": float(st.session_state.get("nr_def_idx_rate", 0.75)),
                            "protect": float(st.session_state.get("nr_def_protect", 0.5)),
                            "volume_envelope": float(st.session_state.get("nr_def_vol_env", 1.0)),
                            "autotune": bool(st.session_state.get("nr_def_autotune", False)),
                            "proposed_pitch": bool(st.session_state.get("nr_def_proposed", False)),
                            "split_audio": bool(st.session_state.get("nr_def_split", False)),
                            "format": str(st.session_state.get("nr_def_fmt", "WAV")).upper(),
                            "model": "",
                            "index": "",
                            "converted_path": "",
                        })
                        st.session_state["nr_inf_log"] = f"Added {_fq_item['title']} following Step 3 settings."
                        st.rerun()

                q_items = st.session_state["nr_sample_queue"]
                if q_items:
                    qh_l, qh_m, qh_r = st.columns([3, 1.8, 1], gap="small")
                    qh_l.markdown(f'<div class="nr-field-title" style="padding-top:8px">Queued Audio Samples ({len(q_items)})</div>', unsafe_allow_html=True)
                    if len(q_items) > 1 and qh_m.button("Apply Step 3 Settings to All", type="secondary", use_container_width=True, key="nr_apply_def_all"):
                        for c in q_items:
                            cid_c = c["id"]
                            c["pitch"] = int(st.session_state.get("nr_def_pitch", 0))
                            c["index_rate"] = float(st.session_state.get("nr_def_idx_rate", 0.75))
                            c["protect"] = float(st.session_state.get("nr_def_protect", 0.5))
                            c["volume_envelope"] = float(st.session_state.get("nr_def_vol_env", 1.0))
                            c["autotune"] = bool(st.session_state.get("nr_def_autotune", False))
                            c["proposed_pitch"] = bool(st.session_state.get("nr_def_proposed", False))
                            c["split_audio"] = bool(st.session_state.get("nr_def_split", False))
                            c["format"] = str(st.session_state.get("nr_def_fmt", "WAV")).upper()
                            for cf in ("custom_pitch", "custom_index", "custom_protect", "custom_vol", "custom_at", "custom_pp", "custom_sp", "custom_fmt"):
                                c[cf] = False
                            st.session_state[f"nr_cp_{cid_c}"] = c["pitch"]
                            st.session_state[f"nr_ci_{cid_c}"] = c["index_rate"]
                            st.session_state[f"nr_cprot_{cid_c}"] = c["protect"]
                            st.session_state[f"nr_cvol_{cid_c}"] = c["volume_envelope"]
                            st.session_state[f"nr_cat_{cid_c}"] = c["autotune"]
                            st.session_state[f"nr_cpp_{cid_c}"] = c["proposed_pitch"]
                            st.session_state[f"nr_csp_{cid_c}"] = c["split_audio"]
                            st.session_state[f"nr_cf_{cid_c}"] = c["format"]
                        st.rerun()
                    if qh_r.button("Clear All", type="secondary", use_container_width=True, key="nr_clear_all_clips"):
                        for c in list(q_items):
                            _clear_clip_widget_keys(c["id"])
                        st.session_state["nr_sample_queue"] = []
                        st.rerun()

                    for idx_c, clip in enumerate(list(q_items)):
                        cid = clip["id"]
                        cp_k = f"nr_cp_{cid}"
                        ci_k = f"nr_ci_{cid}"
                        cm_k = f"nr_cm_{cid}"
                        cx_k = f"nr_cx_{cid}"
                        cf_k = f"nr_cf_{cid}"
                        cprot_k = f"nr_cprot_{cid}"
                        cvol_k = f"nr_cvol_{cid}"
                        cat_k = f"nr_cat_{cid}"
                        cpp_k = f"nr_cpp_{cid}"
                        csp_k = f"nr_csp_{cid}"

                        st.session_state.setdefault(cp_k, int(clip["pitch"]))
                        st.session_state.setdefault(ci_k, float(clip["index_rate"]))
                        st.session_state.setdefault(cm_k, str(clip.get("model", "")))
                        st.session_state.setdefault(cx_k, str(clip.get("index", "")))
                        fmt_cur_u = str(clip.get("format", "WAV")).upper()
                        st.session_state.setdefault(cf_k, fmt_cur_u if fmt_cur_u in ("WAV", "MP3", "FLAC") else "WAV")
                        st.session_state.setdefault(cprot_k, float(clip["protect"]))
                        st.session_state.setdefault(cvol_k, float(clip["volume_envelope"]))
                        st.session_state.setdefault(cat_k, bool(clip["autotune"]))
                        st.session_state.setdefault(cpp_k, bool(clip["proposed_pitch"]))
                        st.session_state.setdefault(csp_k, bool(clip["split_audio"]))

                        with st.container(border=True):
                            ch1, ch2, ch3 = st.columns([4, 1.3, 0.55], gap="small")
                            p_str = f'+{clip["pitch"]}' if clip["pitch"] > 0 else str(clip["pitch"])
                            p_badge = (
                                '<span style="border:1px solid rgba(245,158,11,0.5);background:#1B1207;color:#F59E0B;border-radius:4px;padding:1px 6px;font-family:JetBrains Mono,monospace;font-size:10.5px;font-weight:600">Custom Override</span>'
                                if clip.get("custom_pitch")
                                else '<span style="border:1px solid rgba(16,185,129,0.5);background:#081A14;color:#10B981;border-radius:4px;padding:1px 6px;font-family:JetBrains Mono,monospace;font-size:10.5px;font-weight:600">Following Step 3</span>'
                            )
                            status_tag = " · Converted" if clip.get("converted_path") else ""
                            ch1.markdown(
                                f'<div style="font-family:JetBrains Mono,monospace;font-size:13px;font-weight:600;color:#FDF7F7">{idx_c + 1}. {html.escape(clip["name"])}</div>'
                                f'<div style="font-size:12.5px;color:#C89EA4;display:flex;flex-wrap:wrap;align-items:center;gap:6px;margin-top:2px">'
                                f'<span>Pitch {p_str}</span>{p_badge}<span>· Index {clip["index_rate"]:.2f} · {clip["format"].upper()}{status_tag}</span></div>',
                                unsafe_allow_html=True,
                            )
                            if ch2.button("Re-Convert" if clip.get("converted_path") else "Convert", type="primary", use_container_width=True, key=f"nr_conv_one_{cid}"):
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
                                        st.session_state["nr_inf_log"] = _filter_status_lines(msg, msg)
                                        if rc == 0 and os.path.isfile(out_abs):
                                            clip["converted_path"] = out_abs
                                            st.session_state["nr_saved_samples"].insert(0, {
                                                "id": _next_clip_id("out"),
                                                "name": out_name,
                                                "path": out_abs,
                                                "pitch": int(clip["pitch"]),
                                                "index_rate": float(clip["index_rate"]),
                                                "model": Path(chosen_pth).name,
                                                "created_at": time.strftime("%H:%M:%S"),
                                            })
                                    st.rerun()
                            if ch3.button("✕", key=f"nr_rm_clip_{cid}", help="Discard clip"):
                                st.session_state["nr_sample_queue"].pop(idx_c)
                                _clear_clip_widget_keys(cid)
                                st.rerun()

                            src_abs = Path(now_dir) / clip["path"]
                            if src_abs.is_file():
                                st.audio(_read_file_bytes_cached(str(src_abs), src_abs.stat().st_mtime), format=_audio_mime(src_abs))

                            has_custom = any(bool(clip.get(cf, False)) for cf in ("custom_pitch", "custom_index", "custom_protect", "custom_vol", "custom_at", "custom_pp", "custom_sp", "custom_fmt")) or bool(clip.get("model")) or bool(clip.get("index"))
                            with st.expander("Override pitch, index, model & advanced (optional)", expanded=False):
                                st.caption("Changes here apply only to this clip. Leave untouched to follow Step 3 automatically.")
                                if has_custom and st.button("Use Step 3 Settings", type="secondary", key=f"nr_reset_clip_{cid}"):
                                    clip["pitch"] = int(st.session_state.get("nr_def_pitch", 0))
                                    clip["index_rate"] = float(st.session_state.get("nr_def_idx_rate", 0.75))
                                    clip["protect"] = float(st.session_state.get("nr_def_protect", 0.5))
                                    clip["volume_envelope"] = float(st.session_state.get("nr_def_vol_env", 1.0))
                                    clip["autotune"] = bool(st.session_state.get("nr_def_autotune", False))
                                    clip["proposed_pitch"] = bool(st.session_state.get("nr_def_proposed", False))
                                    clip["split_audio"] = bool(st.session_state.get("nr_def_split", False))
                                    clip["format"] = str(st.session_state.get("nr_def_fmt", "WAV")).upper()
                                    clip["model"] = ""
                                    clip["index"] = ""
                                    for cf in ("custom_pitch", "custom_index", "custom_protect", "custom_vol", "custom_at", "custom_pp", "custom_sp", "custom_fmt"):
                                        clip[cf] = False
                                    st.session_state[cp_k] = clip["pitch"]
                                    st.session_state[ci_k] = clip["index_rate"]
                                    st.session_state[cprot_k] = clip["protect"]
                                    st.session_state[cvol_k] = clip["volume_envelope"]
                                    st.session_state[cat_k] = clip["autotune"]
                                    st.session_state[cpp_k] = clip["proposed_pitch"]
                                    st.session_state[csp_k] = clip["split_audio"]
                                    st.session_state[cf_k] = clip["format"]
                                    st.session_state[cm_k] = ""
                                    st.session_state[cx_k] = ""
                                    st.rerun()
                                cp1, cp2 = st.columns(2, gap="medium")
                                with cp1:
                                    new_cp = int(st.slider("Clip Pitch Override", -32, 32, step=1, key=cp_k))
                                    if new_cp != int(clip["pitch"]):
                                        clip["pitch"] = new_cp
                                        clip["custom_pitch"] = True
                                with cp2:
                                    new_ci = float(st.slider("Clip Index Rate Override", 0.0, 1.0, step=0.05, key=ci_k))
                                    if abs(new_ci - float(clip["index_rate"])) > 1e-6:
                                        clip["index_rate"] = new_ci
                                        clip["custom_index"] = True
                                ov1, ov2, ov3 = st.columns(3, gap="small")
                                clip["model"] = ov1.selectbox("Clip Voice Model", options=[""] + pths, format_func=lambda x: "Use Default (Step 1)" if x == "" else x, key=cm_k)
                                clip["index"] = ov2.selectbox("Clip Index File", options=[""] + idxs, format_func=lambda x: "Use Default / None" if x == "" else x, key=cx_k)
                                new_cf = str(ov3.selectbox("Export Format", ["WAV", "MP3", "FLAC"], key=cf_k)).upper()
                                if new_cf != str(clip["format"]).upper():
                                    clip["format"] = new_cf
                                    clip["custom_fmt"] = True
                                ov4, ov5 = st.columns(2, gap="small")
                                new_cprot = float(ov4.slider("Protect", 0.0, 0.5, step=0.01, key=cprot_k))
                                if abs(new_cprot - float(clip["protect"])) > 1e-6:
                                    clip["protect"] = new_cprot
                                    clip["custom_protect"] = True
                                new_cvol = float(ov5.slider("Volume Envelope", 0.0, 1.0, step=0.05, key=cvol_k))
                                if abs(new_cvol - float(clip["volume_envelope"])) > 1e-6:
                                    clip["volume_envelope"] = new_cvol
                                    clip["custom_vol"] = True
                                t1, t2, t3 = st.columns(3, gap="small")
                                new_cat = bool(t1.checkbox("Autotune", key=cat_k))
                                if new_cat != bool(clip["autotune"]):
                                    clip["autotune"] = new_cat
                                    clip["custom_at"] = True
                                new_cpp = bool(t2.checkbox("Proposed Pitch", key=cpp_k))
                                if new_cpp != bool(clip["proposed_pitch"]):
                                    clip["proposed_pitch"] = new_cpp
                                    clip["custom_pp"] = True
                                new_csp = bool(t3.checkbox("Split", key=csp_k))
                                if new_csp != bool(clip["split_audio"]):
                                    clip["split_audio"] = new_csp
                                    clip["custom_sp"] = True

                            if clip.get("converted_path") and os.path.isfile(clip["converted_path"]):
                                conv_p = Path(clip["converted_path"])
                                st.markdown(f'<div style="font-size:12.5px;font-weight:600;color:#FDF7F7;margin-top:6px">Converted Output ({html.escape(conv_p.name)})</div>', unsafe_allow_html=True)
                                conv_bytes = _read_file_bytes_cached(str(conv_p), conv_p.stat().st_mtime)
                                st.audio(conv_bytes, format=_audio_mime(conv_p))
                                st.download_button("Save Converted Clip", data=conv_bytes, file_name=conv_p.name, type="secondary", key=f"nr_dl_c_{cid}")
            else:
                st.session_state.setdefault("nr_batch_folder_inp", st.session_state.get("nr_batch_folder", "/kaggle/working/audio_inputs"))
                st.session_state["nr_batch_folder"] = st.text_input(
                    "Input Folder Path (Batch)",
                    key="nr_batch_folder_inp",
                )

        # --- STEP 3: Pitch & Settings ---
        with _nr_card("nr_samples_step_3"):
            _step_head(3, "Pitch & Settings", "Master Voice Settings — controls Pitch and Voice Similarity for all clips unless overridden in Step 2.")

            st.session_state.setdefault("nr_def_pitch", 0)
            st.session_state["nr_def_pitch"] = _synced_slider_num(
                "Pitch",
                "Shift pitch in semitones (-32 to +32). Use +12 for male to female, -12 for female to male, or 0 to match original pitch.",
                "nr_def_pitch",
                -32,
                32,
                0,
            )

            _pp_cols = st.columns(5, gap="small")
            for _pp_idx, (_pp_lbl, _pp_val) in enumerate((("-12 F → M", -12), ("-5 Lower", -5), ("0 Same Pitch", 0), ("+5 Higher", 5), ("+12 M → F", 12))):
                if _pp_cols[_pp_idx].button(
                    _pp_lbl,
                    type="primary" if int(st.session_state.get("nr_def_pitch", 0)) == _pp_val else "secondary",
                    use_container_width=True,
                    key=f"nr_pitch_pre_{_pp_val}",
                ):
                    st.session_state["nr_def_pitch"] = _pp_val
                    st.session_state["nr_def_pitch__sl"] = _pp_val
                    st.session_state["nr_def_pitch__nm"] = _pp_val
                    st.rerun()

            with st.container(border=True):
                st.markdown('<div class="nr-field-title">Voice Similarity (Index Rate)</div><div class="nr-field-help">Controls how strongly converted vocals adopt your trained .index accent &amp; timbre (0.75 recommended).</div>', unsafe_allow_html=True)
                st.session_state["nr_def_idx_rate"] = st.slider("Voice Similarity (Index Rate)", 0.0, 1.0, step=0.05, label_visibility="collapsed", key="nr_adv_idx")

            f0_cur = st.session_state.get("nr_def_f0", "rmvpe")
            idx_cur = float(st.session_state.get("nr_def_idx_rate", 0.75))
            env_cur = float(st.session_state.get("nr_def_vol_env", 1.0))
            prot_cur = float(st.session_state.get("nr_def_protect", 0.5))
            adv_lbl = f"Advanced Settings ({f0_cur} · index {idx_cur:.2f} · env {env_cur:g} · protect {prot_cur:g})"
            with st.expander(adv_lbl, expanded=False):
                with st.expander("Voice Matching", expanded=True):
                    st.session_state["nr_def_vol_env"] = st.slider("Volume Envelope", 0.0, 1.0, step=0.05, key="nr_adv_vol")
                    st.session_state["nr_def_protect"] = st.slider("Protect Voiceless Consonants", 0.0, 0.5, step=0.01, key="nr_adv_prot")

                with st.expander("Models & Algorithms", expanded=False):
                    ma1, ma2, ma3 = st.columns(3, gap="small")
                    st.session_state["nr_def_sid"] = ma1.number_input("Speaker ID", 0, 100, step=1, key="nr_adv_sid")
                    st.session_state["nr_def_f0"] = ma2.selectbox("Pitch extraction algorithm", ["rmvpe", "fcpe", "swift-f0", "crepe"], key="nr_adv_f0")
                    st.session_state["nr_def_emb"] = ma3.selectbox("Embedder Model", ["contentvec", "spin", "chinese-hubert-base"], key="nr_adv_emb")

                with st.expander("Advanced Pitch & Tuning", expanded=False):
                    pt1, pt2, pt3 = st.columns(3, gap="small")
                    st.session_state["nr_def_autotune"] = pt1.checkbox("Autotune", key="nr_adv_at")
                    st.session_state["nr_def_proposed"] = pt2.checkbox("Proposed Pitch", key="nr_adv_pp")
                    st.session_state["nr_def_split"] = pt3.checkbox("Split Audio", key="nr_adv_sp")

                with st.expander("Audio Cleanup", expanded=False):
                    st.session_state["nr_def_clean"] = st.checkbox("Audio cleanup (balanced rumble filter & gentle de-noise)", key="nr_adv_clean")
                    st.session_state["nr_def_clean_db"] = st.slider("Noise reduction limit (dB)", 1, 30, step=1, disabled=not bool(st.session_state["nr_def_clean"]), key="nr_adv_cleandb")

                with st.expander("Formant Shifting", expanded=False):
                    st.session_state["nr_def_formant"] = st.checkbox("Enable Formant Shifting", key="nr_adv_fmt_on")
                    if st.session_state["nr_def_formant"]:
                        fs1, fs2 = st.columns(2, gap="small")
                        st.session_state["nr_def_qf"] = fs1.slider("Formant Quefrency", 0.0, 16.0, step=0.1, key="nr_adv_qf")
                        st.session_state["nr_def_tf"] = fs2.slider("Formant Timbre", 0.0, 16.0, step=0.1, key="nr_adv_tf")

                with st.expander("Output Settings", expanded=False):
                    os1, os2 = st.columns([3, 1.2], gap="small")
                    st.session_state["nr_def_out_path"] = os1.text_input("Output Path", key="nr_adv_outp")
                    st.session_state["nr_def_fmt"] = os2.selectbox("Export Format", ["WAV", "MP3", "FLAC"], key="nr_adv_fmt")
                    if st.button("Clear Outputs (Deletes all audios in assets/audios)", type="secondary", use_container_width=True, key="nr_btn_clr_out"):
                        au_dir = Path(now_dir) / "assets" / "audios"
                        if au_dir.is_dir():
                            for f in au_dir.iterdir():
                                if f.is_file():
                                    f.unlink(missing_ok=True)
                        for c_out in st.session_state["nr_sample_queue"]:
                            c_out["converted_path"] = ""
                        st.session_state["nr_saved_samples"] = []
                        st.session_state["nr_inf_log"] = "Cleared all generated outputs in assets/audios."
                        _clear_fs_caches()
                        try:
                            st.rerun(scope="app")
                        except TypeError:
                            st.rerun()

        # --- STEP 4: Convert & Output ---
        with _nr_card("nr_samples_step_4"):
            _step_head(4, "Convert & Output", "Render your queued recordings or batch folder, then listen and download results below.")

            with st.container(border=True):
                st.session_state.setdefault("nr_inf_terms", True)
                inf_terms = st.checkbox("I agree to the terms of use — Confirm responsible usage guidelines prior to rendering voice conversion samples.", key="nr_inf_terms")
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
                        st.session_state["nr_inf_log"] = _filter_status_lines(msg, msg)
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
                            msgs.append(f"{clip['name']}: {_filter_status_lines(msg, msg)}")
                            if rc == 0 and os.path.isfile(out_abs):
                                clip["converted_path"] = out_abs
                                st.session_state["nr_saved_samples"].insert(0, {
                                    "id": _next_clip_id("out"),
                                    "name": out_name,
                                    "path": out_abs,
                                    "pitch": int(clip["pitch"]),
                                    "index_rate": float(clip["index_rate"]),
                                    "model": Path(c_pth).name,
                                    "created_at": time.strftime("%H:%M:%S"),
                                })
                        st.session_state["nr_inf_log"] = "\\n".join(msgs)
                    st.rerun()

            _output_box("Sample output", _filter_status_lines(st.session_state["nr_inf_log"], st.session_state["nr_inf_log"]))

            saved_list = st.session_state.get("nr_saved_samples", [])
            if saved_list:
                with st.container(border=True):
                    st.markdown(f'<div class="nr-field-title">Recent Generated Samples ({len(saved_list)})</div>', unsafe_allow_html=True)
                    for s_item in list(saved_list[:12]):
                        s_id = str(s_item["id"])
                        with st.container(border=True):
                            sr1, sr2, sr3 = st.columns([4, 1.1, 0.5], gap="small")
                            sp_str = f'+{s_item["pitch"]}' if s_item["pitch"] > 0 else str(s_item["pitch"])
                            sr1.markdown(
                                f'<div style="font-family:JetBrains Mono,monospace;font-size:12px;color:#FDF7F7">{html.escape(s_item["name"])}</div>'
                                f'<div style="font-size:11px;color:#AD8288">{html.escape(s_item["model"])} · Pitch {sp_str} · idx {s_item["index_rate"]:.2f} · {s_item["created_at"]}</div>',
                                unsafe_allow_html=True,
                            )
                            if os.path.isfile(s_item["path"]):
                                sp_obj = Path(s_item["path"])
                                sp_bytes = _read_file_bytes_cached(str(sp_obj), sp_obj.stat().st_mtime)
                                sr2.download_button("Save", data=sp_bytes, file_name=s_item["name"], type="secondary", use_container_width=True, key=f"nr_dl_sav_{s_id}")
                                st.audio(sp_bytes, format=_audio_mime(sp_obj))
                            if sr3.button("✕", key=f"nr_rm_sav_{s_id}"):
                                st.session_state["nr_saved_samples"] = [x for x in st.session_state["nr_saved_samples"] if x.get("id") != s_id]
                                st.rerun()

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
            with st.expander("View Generated core.py CLI Command", expanded=False):
                st.code(cli_cmd, language="bash")

    if _HAS_FRAGMENT:
        @st.fragment
        def _render_samples_workspace_frag() -> None:
            _render_samples_workspace_body()

        _render_samples_workspace_frag()
    else:
        _render_samples_workspace_body()

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
                    "id": _next_clip_id(f"fs-{item['id']}"),
                    "name": item["fileName"],
                    "path": rel_p,
                    "pitch": int(st.session_state.get("nr_def_pitch", 0)),
                    "index_rate": float(st.session_state.get("nr_def_idx_rate", 0.75)),
                    "protect": float(st.session_state.get("nr_def_protect", 0.5)),
                    "volume_envelope": float(st.session_state.get("nr_def_vol_env", 1.0)),
                    "autotune": bool(st.session_state.get("nr_def_autotune", False)),
                    "proposed_pitch": bool(st.session_state.get("nr_def_proposed", False)),
                    "split_audio": bool(st.session_state.get("nr_def_split", False)),
                    "format": str(st.session_state.get("nr_def_fmt", "WAV")).upper(),
                    "model": "",
                    "index": "",
                    "converted_path": "",
                })
                queued_paths.add(rel_p)
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
                                "id": _next_clip_id(f"fs-{sample['id']}"),
                                "name": sample["fileName"],
                                "path": str(local_wav.relative_to(now_dir)),
                                "pitch": int(st.session_state.get("nr_def_pitch", 0)),
                                "index_rate": float(st.session_state.get("nr_def_idx_rate", 0.75)),
                                "protect": float(st.session_state.get("nr_def_protect", 0.5)),
                                "volume_envelope": float(st.session_state.get("nr_def_vol_env", 1.0)),
                                "autotune": bool(st.session_state.get("nr_def_autotune", False)),
                                "proposed_pitch": bool(st.session_state.get("nr_def_proposed", False)),
                                "split_audio": bool(st.session_state.get("nr_def_split", False)),
                                "format": str(st.session_state.get("nr_def_fmt", "WAV")).upper(),
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
    if parsed_diag and m_diag_raw and not st.session_state.get("nr_diag_log_input", "").strip():
        tb_m = _read_tb_metrics_for_model(m_diag_raw)
        for tb_k in ("avg_kl_loss", "avg_fm_loss", "avg_disc_loss", "avg_gen_loss", "latest_grad_norm"):
            if parsed_diag.get(tb_k) is None and tb_m.get(tb_k) is not None:
                parsed_diag[tb_k] = tb_m[tb_k]
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
            _v_kind = str(zone.get("verdict", "sweet_spot"))
            _v_border = "#E51923" if _v_kind == "overtrained" else ("rgba(245,158,11,0.75)" if _v_kind == "undertrained" else "rgba(16,185,129,0.65)")
            _v_bg = "#24090D" if _v_kind == "overtrained" else ("#1B1207" if _v_kind == "undertrained" else "#081A14")
            _under_pct = int(zone.get("under_risk_pct", 10))
            _over_pct = int(zone.get("over_risk_pct", 10))
            _spc = zone.get("steps_per_clip")
            _eff_steps = int(zone.get("effective_steps", 0))
            _spc_badge = f'<span class="nr-mono-badge">{_spc} steps/clip</span>' if _spc is not None else ""
            st.markdown(
                f'<div style="border:1px solid {_v_border};background:{_v_bg};border-radius:10px;padding:14px 16px;margin:10px 0">'
                f'<div style="display:flex;flex-wrap:wrap;justify-content:space-between;align-items:flex-start;gap:10px">'
                f'<div><div style="font-family:Space Grotesk,sans-serif;font-size:17px;font-weight:700;color:#FDF7F7">{html.escape(str(zone.get("verdict_title", "Optimal Sweet Spot")))}</div>'
                f'<div style="font-size:13px;color:#DFBFC3;margin-top:3px">{html.escape(str(zone.get("verdict_guidance", "")))}</div></div>'
                f'<div style="display:flex;gap:8px;flex-wrap:wrap">'
                f'{_spc_badge}'
                f'<span class="nr-mono-badge">{_eff_steps} eff. steps</span></div></div></div>',
                unsafe_allow_html=True,
            )
            _rm1, _rm2 = st.columns(2, gap="small")
            _rm1.markdown(
                f'<div class="nr-metric-box" style="margin-bottom:8px">'
                f'<div style="display:flex;justify-content:space-between;font-size:12px;margin-bottom:6px">'
                f'<span style="color:#F59E0B;font-weight:600">Undertrained Risk</span>'
                f'<span style="font-family:JetBrains Mono,monospace;color:#FDF7F7;font-weight:700">{_under_pct}%</span></div>'
                f'<div style="height:8px;background:#160709;border:1px solid #47161D;border-radius:3px;overflow:hidden">'
                f'<div style="height:100%;width:{max(0, min(100, _under_pct))}%;background:#F59E0B"></div></div></div>',
                unsafe_allow_html=True,
            )
            _rm2.markdown(
                f'<div class="nr-metric-box" style="margin-bottom:8px">'
                f'<div style="display:flex;justify-content:space-between;font-size:12px;margin-bottom:6px">'
                f'<span style="color:#FF5A62;font-weight:600">Overtrained Risk</span>'
                f'<span style="font-family:JetBrains Mono,monospace;color:#FDF7F7;font-weight:700">{_over_pct}%</span></div>'
                f'<div style="height:8px;background:#160709;border:1px solid #47161D;border-radius:3px;overflow:hidden">'
                f'<div style="height:100%;width:{max(0, min(100, _over_pct))}%;background:#E51923"></div></div></div>',
                unsafe_allow_html=True,
            )

            cand_badges = " ".join(
                f'<span class="nr-mono-badge" style="{"border-color:#E51923;color:#FDF7F7;background:#2A0D12;" if ep_c == zone["best_epoch"] else ""}">Epoch {ep_c}</span>'
                for ep_c in zone["candidate_epochs"]
            )
            best_loss_lbl = f"{zone['best_loss']:.3f}" if zone["best_loss"] is not None else "—"
            ds_match_pct = int(zone.get("dataset_match_pct", 0))
            stab_pct = int(zone.get("stability_pct", 0))
            conv_lbl = str(zone.get("convergence_label", "Converging"))
            with st.container(border=True):
                st.markdown(
                    f'<div style="display:flex;flex-wrap:wrap;justify-content:space-between;align-items:center;gap:12px;margin-bottom:10px">'
                    f'<div><div style="font-size:12.5px;font-weight:600;color:#FF5A63;text-transform:uppercase;letter-spacing:0.04em">Recommended Best Epoch</div>'
                    f'<div style="display:flex;align-items:baseline;gap:14px;flex-wrap:wrap;margin-top:2px">'
                    f'<span style="font-family:Space Grotesk,sans-serif;font-size:28px;font-weight:700;color:#FDF7F7;line-height:1.15">Epoch {zone["best_epoch"]}</span>'
                    f'<span class="nr-mono-badge" style="border-color:#E51923;color:#FDF7F7">Dataset Match: {ds_match_pct}% · Stability {stab_pct}% · {html.escape(conv_lbl)}</span></div>'
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
                    f'<div class="nr-zone-card nr-zone-sweet"><div style="font-size:12.5px;font-weight:600;color:#FDF7F7">Dataset Closeness &amp; Window</div>'
                    f'<div style="font-family:JetBrains Mono,monospace;font-size:16px;font-weight:700;color:#FDF7F7;margin-top:4px">{ds_match_pct}% Fit · e{zone["sweet_min"]}–e{zone["sweet_max"]}</div>'
                    f'<div style="font-size:12.5px;color:#DFBFC3;margin-top:4px">Test saved checkpoints in Samples tab.</div></div>',
                    unsafe_allow_html=True,
                )
                zc3.markdown(
                    f'<div class="nr-zone-card"><div style="font-size:12.5px;font-weight:600;color:#FF2B35">Overtraining Watch</div>'
                    f'<div style="font-family:JetBrains Mono,monospace;font-size:16px;font-weight:700;color:#FDF7F7;margin-top:4px">{zone["over_label"]}</div>'
                    f'<div style="font-size:12.5px;color:#C89EA4;margin-top:4px">Watch for metallic timbre past this point.</div></div>',
                    unsafe_allow_html=True,
                )

                breakdown_items = zone.get("metric_breakdown") or []
                if breakdown_items:
                    st.markdown('<div class="nr-field-title" style="margin:12px 0 6px 0">Multi-Metric &amp; Dataset Fit Breakdown</div>', unsafe_allow_html=True)
                    for row_start in range(0, len(breakdown_items), 4):
                        mb_cols = st.columns(4, gap="small")
                        for c_idx, mb_item in enumerate(breakdown_items[row_start:row_start + 4]):
                            st_c = "#10B981" if mb_item.get("status") == "optimal" else ("#F59E0B" if mb_item.get("status") == "watch" else "#FF2B35")
                            mb_cols[c_idx].markdown(
                                f'<div class="nr-metric-box" style="margin-bottom:6px">'
                                f'<div style="display:flex;justify-content:space-between;align-items:center">'
                                f'<span class="nr-metric-k">{html.escape(str(mb_item["label"]))}</span>'
                                f'<span style="width:7px;height:7px;border-radius:99px;background:{st_c};display:inline-block"></span></div>'
                                f'<div class="nr-metric-v">{html.escape(str(mb_item["value"]))}</div>'
                                f'<div style="font-size:11.5px;color:#C89EA4;margin-top:3px">{html.escape(str(mb_item["detail"]))}</div></div>',
                                unsafe_allow_html=True,
                            )

                ranked_ckpts = zone.get("ranked_checkpoints") or []
                if ranked_ckpts:
                    st.markdown('<div class="nr-field-title" style="margin:10px 0 6px 0">Top Recommended Checkpoints — 1-Click Audition</div>', unsafe_allow_html=True)
                    _avail_pths = _list_pth_files("")
                    _m_slug = _safe_model_slug(m_diag_raw)
                    for _rk_i, _rk in enumerate(ranked_ckpts):
                        _rk_ep = int(_rk["epoch"])
                        _rk_cols = st.columns([0.5, 1.8, 1.0, 1.1, 1.35], gap="small")
                        _rk_cols[0].markdown(f'<div style="font-family:JetBrains Mono,monospace;font-size:12.5px;font-weight:700;color:#FDF7F7;padding-top:6px">#{_rk_i + 1}</div>', unsafe_allow_html=True)
                        _rk_cols[1].markdown(
                            f'<div style="font-family:JetBrains Mono,monospace;font-size:12.5px;color:#FDF7F7">{html.escape(str(_rk["checkpoint"]))}</div>'
                            f'<div style="font-size:11px;color:#C89EA4">{html.escape(str(_rk["tag"]))} · Score {_rk["score"]}/100</div>',
                            unsafe_allow_html=True,
                        )
                        _rk_cols[2].markdown(f'<div style="font-family:JetBrains Mono,monospace;font-size:12.5px;color:#DFBFC3;padding-top:6px">Mel {float(_rk["mel_loss"]):.3f}</div>', unsafe_allow_html=True)
                        _rk_cols[3].markdown(f'<div style="font-family:JetBrains Mono,monospace;font-size:12px;color:#C89EA4;padding-top:6px">+{float(_rk["drift_pct"]):.1f}% drift</div>', unsafe_allow_html=True)
                        if _rk_cols[4].button("Load in Samples", type="primary" if _rk.get("status") == "best" else "secondary", use_container_width=True, key=f"nr_diag_load_{_rk_ep}"):
                            _cand_match = next((p for p in _avail_pths if f"_{_rk_ep}e" in Path(p).name and (not m_diag_raw or _m_slug.lower() in p.lower())), "")
                            if not _cand_match:
                                _cand_match = next((p for p in _avail_pths if f"_{_rk_ep}e" in Path(p).name), "")
                            _chosen_p = _cand_match or (f"logs/{_m_slug}/{_rk['checkpoint']}" if m_diag_raw else str(_rk["checkpoint"]))
                            st.session_state["nr_inf_pth"] = _chosen_p
                            _auto_i = _match_index(_chosen_p, _list_index_files(""))
                            if _auto_i:
                                st.session_state["nr_inf_idx"] = _auto_i
                            st.session_state["nr_active_tab"] = "Samples"
                            _push_toast("Checkpoint Loaded", f"Selected {Path(_chosen_p).name} in Samples.")
                            st.rerun()

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
                coords = [(to_x(p["epoch"]), PAD_T + (1.0 - (p["lowest_loss"] - min_l) / span_l) * PLOT_H, p["epoch"], float(p["lowest_loss"])) for p in pts]
                poly_pts = " ".join(f"{cx:.1f},{cy:.1f}" for cx, cy, _, _ in coords)
                sw_s = to_x(zone["sweet_min"])
                sw_e = to_x(zone["sweet_max"])
                best_x = to_x(zone["best_epoch"])
                ov_s = to_x(zone["over_start"]) if zone.get("over_start") else W - PAD_R
                ov_w = max(0.0, (W - PAD_R) - ov_s)
                ov_rect = f'<rect x="{ov_s:.1f}" y="{PAD_T}" width="{ov_w:.1f}" height="{PLOT_H}" fill="#991B1B" fill-opacity="0.22" />' if ov_w > 4 else ""
                dots_svg = "".join(
                    f'<g><circle cx="{cx:.1f}" cy="{cy:.1f}" r="3.5" fill="#FDF7F7" stroke="#E51923" stroke-width="2"><title>Epoch {ep_v} · Mel {mel_v:.3f}</title></circle>'
                    f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="9" fill="transparent"><title>Epoch {ep_v} · Mel {mel_v:.3f}</title></circle></g>'
                    for cx, cy, ep_v, mel_v in coords
                )
                lbl_coords = [coords[0]] if coords else []
                for cx, cy, ep_v, mel_v in coords[1:-1]:
                    if cx - lbl_coords[-1][0] >= 55 and coords[-1][0] - cx >= 55:
                        lbl_coords.append((cx, cy, ep_v, mel_v))
                if len(coords) > 1:
                    lbl_coords.append(coords[-1])
                lbls_svg = "".join(f'<text x="{cx:.1f}" y="{H - 5}" text-anchor="middle" fill="#C89EA4" font-size="11" font-family="JetBrains Mono, monospace">e{ep_v}</text>' for cx, _, ep_v, _ in lbl_coords)
                grid_svg = "".join(f'<line x1="{PAD_L}" y1="{PAD_T + PLOT_H * r:.1f}" x2="{W - PAD_R}" y2="{PAD_T + PLOT_H * r:.1f}" stroke="#47161D" stroke-dasharray="3 3" />' for r in (0.25, 0.5, 0.75))

                svg_html = (
                    f'<div style="background:#160709;border:1px solid #47161D;border-radius:8px;padding:12px;margin:10px 0">'
                    f'<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px">'
                    f'<span style="font-size:13px;font-weight:600;color:#FDF7F7">Observed Mel Loss &amp; Best Epoch (e{zone["best_epoch"]})</span>'
                    f'<span style="font-family:JetBrains Mono,monospace;font-size:11px;color:#C89EA4">● Best Epoch Window</span></div>'
                    f'<svg viewBox="0 0 {W} {H}" style="width:100%;height:auto">'
                    f'<rect x="{sw_s:.1f}" y="{PAD_T}" width="{max(8.0, sw_e - sw_s):.1f}" height="{PLOT_H}" fill="#E51923" fill-opacity="0.16" />'
                    f'{ov_rect}'
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
            st.session_state.setdefault("nr_vb_ratio", 0.5)
            st.session_state.setdefault("nr_vb_ratio_sl", float(st.session_state["nr_vb_ratio"]))
            cur_vb_ratio = float(st.session_state.get("nr_vb_ratio_sl", 0.5))
            br_l, br_val, br_rst = st.columns([4.4, 0.85, 0.55], gap="small")
            br_l.markdown('<div class="nr-field-title">Blend Ratio</div><div class="nr-field-help">Balance influence between checkpoints: 0 leans 100% toward Model B, while 1 leans 100% toward Model A.</div>', unsafe_allow_html=True)
            br_val.markdown(f'<div style="background:#160709;border:1px solid #47161D;border-radius:8px;padding:6px 10px;text-align:center;font-family:JetBrains Mono,monospace;font-size:13px;color:#FDF7F7">{cur_vb_ratio:g}</div>', unsafe_allow_html=True)
            if br_rst.button("↺", key="nr_vb_rst_ratio", help="Reset Blend Ratio to 0.5", use_container_width=True):
                st.session_state["nr_vb_ratio"] = 0.5
                st.session_state["nr_vb_ratio_sl"] = 0.5
                st.rerun()
            blend_ratio = st.slider("Blend Ratio", 0.0, 1.0, step=0.05, label_visibility="collapsed", key="nr_vb_ratio_sl")
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
            rel_link = str(st.session_state.get("nr_rel_url") or "")
            if rel_link and ("Update available" in str(st.session_state.get("nr_ver_status") or "")):
                st.link_button("Open GitHub Releases", rel_link, type="secondary")
            def _ver_tuple(ver_s: str) -> tuple[int, ...]:
                nums = re.findall(r"[0-9]+", str(ver_s or ""))
                return tuple(int(x) for x in nums[:4]) if nums else (0,)
            def _is_newer_ver(cand_s: str, cur_s: str) -> bool:
                a = _ver_tuple(cand_s)
                b = _ver_tuple(cur_s)
                mlen = max(len(a), len(b))
                a_pad = a + (0,) * (mlen - len(a))
                b_pad = b + (0,) * (mlen - len(b))
                return a_pad > b_pad
            def _extract_rev_and_ver(raw_bytes: bytes) -> tuple[int, str]:
                txt_b = raw_bytes.decode("utf-8", errors="replace")
                m_rev = re.search(r"NR_BUILD_REV\\s*=\\s*([0-9]+)", txt_b)
                m_ver = re.search(r"""APP_VERSION\\s*=\\s*['"]([0-9]+\\.[0-9]+\\.[0-9]+)['"]""", txt_b)
                return (int(m_rev.group(1)) if m_rev else 0), (m_ver.group(1) if m_ver else "")
            def _trigger_webui_restart(status_msg: str) -> None:
                sup_env = os.environ.get("NR_KAGGLE_SUPERVISOR") == "1"
                has_kaggle_logs = any((Path(now_dir) / fn).is_file() for fn in ("webui-cf.log", "webui-hrzn.log", "cloudflared.log"))
                if sup_env or has_kaggle_logs:
                    NR_STATE_DIR.mkdir(parents=True, exist_ok=True)
                    (NR_STATE_DIR / "nr_restart_webui").write_text("1", encoding="utf-8")
                    if not sup_env:
                        try:
                            leg_logs = Path(now_dir) / "logs"
                            leg_logs.mkdir(parents=True, exist_ok=True)
                            (leg_logs / ".nr_restart_webui").write_text("1", encoding="utf-8")
                        except Exception:
                            pass
                    st.session_state["nr_ver_status"] = status_msg
                    try:
                        import streamlit.components.v1 as _st_comp
                        _st_comp.html(
                            "<script>(function(){var n=0;function ping(){n++;fetch(window.location.pathname||'/',{cache:'no-store'}).then(function(r){if(r.ok||n>=20){try{window.top.location.reload();}catch(e){window.location.reload();}}else{setTimeout(ping,900);}}).catch(function(){if(n>=20){try{window.top.location.reload();}catch(e){window.location.reload();}}else{setTimeout(ping,900);}});}setTimeout(ping,2800);})();</script>",
                            height=0,
                        )
                    except Exception:
                        pass
                    threading.Timer(2.0, lambda: os._exit(0)).start()
                else:
                    st.cache_data.clear()
                    st.cache_resource.clear()
                    st.session_state["nr_ver_status"] = status_msg + " (Standalone mode: cache cleared; reload browser if needed.)"

            u1, u2, u3, _ = st.columns([1.45, 1.85, 1.35, 1.4], gap="small")
            if u1.button(_tr("Check for updates"), type="secondary", use_container_width=True, key="nr_set_chk_upd"):
                gh_headers = {"User-Agent": "NoRegretsRVC-UpdateChecker", "Accept": "application/vnd.github+json", "Cache-Control": "no-cache"}
                newest_ver = APP_VERSION
                rel_url = "https://github.com/HunterThatsMe/NoRegretsRVC/releases"
                sha = ""
                remote_rev = 0
                try:
                    found_rel = False
                    try:
                        req_rel = urllib.request.Request("https://api.github.com/repos/HunterThatsMe/NoRegretsRVC/releases/latest", headers=gh_headers)
                        with urllib.request.urlopen(req_rel, timeout=7) as r_rel:
                            rel_d = json.loads(r_rel.read().decode("utf-8"))
                            tag_s = str(rel_d.get("tag_name") or "").lstrip("vV").strip()
                            if tag_s:
                                newest_ver = tag_s
                                found_rel = True
                            if rel_d.get("html_url"):
                                rel_url = str(rel_d["html_url"])
                    except Exception:
                        pass
                    if not found_rel:
                        try:
                            req_rels = urllib.request.Request("https://api.github.com/repos/HunterThatsMe/NoRegretsRVC/releases?per_page=10", headers=gh_headers)
                            with urllib.request.urlopen(req_rels, timeout=7) as r_rels:
                                rel_list = json.loads(r_rels.read().decode("utf-8"))
                                if isinstance(rel_list, list):
                                    for rel_item in rel_list:
                                        if rel_item.get("draft"):
                                            continue
                                        tag_s = str(rel_item.get("tag_name") or "").lstrip("vV").strip()
                                        if tag_s and (_is_newer_ver(tag_s, newest_ver) or not found_rel):
                                            newest_ver = tag_s
                                            if rel_item.get("html_url"):
                                                rel_url = str(rel_item["html_url"])
                                            found_rel = True
                        except Exception:
                            pass
                    if not found_rel:
                        try:
                            req_tags = urllib.request.Request("https://api.github.com/repos/HunterThatsMe/NoRegretsRVC/tags?per_page=10", headers=gh_headers)
                            with urllib.request.urlopen(req_tags, timeout=7) as r_tags:
                                t_list = json.loads(r_tags.read().decode("utf-8"))
                                if isinstance(t_list, list):
                                    for t_item in t_list:
                                        raw_tag = str(t_item.get("name") or "").strip()
                                        tag_s = raw_tag.lstrip("vV").strip()
                                        if tag_s and (_is_newer_ver(tag_s, newest_ver) or not found_rel):
                                            newest_ver = tag_s
                                            rel_url = f"https://github.com/HunterThatsMe/NoRegretsRVC/releases/tag/{raw_tag}"
                                            found_rel = True
                        except Exception:
                            pass
                    try:
                        req_cm = urllib.request.Request("https://api.github.com/repos/HunterThatsMe/NoRegretsRVC/commits/main", headers=gh_headers)
                        with urllib.request.urlopen(req_cm, timeout=7) as r_cm:
                            cm_d = json.loads(r_cm.read().decode("utf-8"))
                            sha = str(cm_d.get("sha") or "")[:7]
                    except Exception:
                        pass
                    ts_bust = int(time.time())
                    try:
                        req_def = urllib.request.Request(f"https://raw.githubusercontent.com/HunterThatsMe/NoRegretsRVC/main/src/data/defaults.ts?t={ts_bust}", headers={"User-Agent": "NoRegretsRVC", "Cache-Control": "no-cache"})
                        with urllib.request.urlopen(req_def, timeout=6) as r_def:
                            _, raw_ver = _extract_rev_and_ver(r_def.read())
                            if raw_ver and _is_newer_ver(raw_ver, newest_ver):
                                newest_ver = raw_ver
                    except Exception:
                        pass
                    try:
                        req_p1 = urllib.request.Request(f"https://raw.githubusercontent.com/HunterThatsMe/NoRegretsRVC/main/src/utils/streamlitWebuiPart1.ts?t={ts_bust}", headers={"User-Agent": "NoRegretsRVC", "Cache-Control": "no-cache"})
                        with urllib.request.urlopen(req_p1, timeout=6) as r_p1:
                            r_rev, _ = _extract_rev_and_ver(r_p1.read())
                            remote_rev = max(remote_rev, r_rev)
                    except Exception:
                        pass
                    if remote_rev == 0:
                        try:
                            req_app = urllib.request.Request(f"https://raw.githubusercontent.com/HunterThatsMe/NoRegretsRVC/main/streamlit_app.py?t={ts_bust}", headers={"User-Agent": "NoRegretsRVC", "Cache-Control": "no-cache"})
                            with urllib.request.urlopen(req_app, timeout=6) as r_app:
                                r_rev2, r_ver2 = _extract_rev_and_ver(r_app.read())
                                remote_rev = max(remote_rev, r_rev2)
                                if r_ver2 and _is_newer_ver(r_ver2, newest_ver):
                                    newest_ver = r_ver2
                        except Exception:
                            pass
                    st.session_state["nr_rel_url"] = rel_url
                    sha_part = f" (commit {sha})" if sha else ""
                    if _is_newer_ver(newest_ver, APP_VERSION):
                        st.session_state["nr_ver_status"] = f"Update available! Installed: Beta version {APP_VERSION} -> Newest GitHub release: {newest_ver}{sha_part}. Click 'Apply WebUI Update & Restart' to install it."
                    elif remote_rev > NR_BUILD_REV:
                        st.session_state["nr_ver_status"] = f"Update available! A newer WebUI build (rev {remote_rev}{sha_part}) is on GitHub for Beta version {APP_VERSION}. Click 'Apply WebUI Update & Restart' to install it."
                    else:
                        st.session_state["nr_ver_status"] = f"✔ NoRegretsRVC Beta version {APP_VERSION} is up to date (newest version: {newest_ver}){sha_part}."
                except Exception as exc:
                    st.session_state["nr_ver_status"] = f"NoRegretsRVC Beta version {APP_VERSION} (could not reach GitHub: {exc})."
                st.rerun()
            if u2.button("Apply WebUI Update & Restart", type="primary", use_container_width=True, key="nr_set_apply_upd"):
                import base64 as _upd_b64
                upd_errors: list[str] = []
                candidates: dict[str, tuple[Path, str, bytes]] = {}
                ts_bust = int(time.time())
                raw_targets = [
                    ("streamlit_app.py", Path(now_dir) / "streamlit_app.py", "CRIMSON_CSS"),
                    ("noregrets_turbo.py", Path(now_dir) / "noregrets_turbo.py", "def apply("),
                    (".streamlit/config.toml", Path(now_dir) / ".streamlit" / "config.toml", "primaryColor"),
                ]
                for gh_path, dst_path, marker_str in raw_targets:
                    try:
                        req_raw = urllib.request.Request(
                            f"https://raw.githubusercontent.com/HunterThatsMe/NoRegretsRVC/main/{gh_path}?t={ts_bust}",
                            headers={"User-Agent": "NoRegretsRVC", "Cache-Control": "no-cache"},
                        )
                        try:
                            with urllib.request.urlopen(req_raw, timeout=10) as r_up:
                                body_bytes = r_up.read()
                        except Exception:
                            req_alt = urllib.request.Request(
                                f"https://raw.githubusercontent.com/HunterThatsMe/NoRegretsRVC/main/assets/{gh_path}?t={ts_bust}",
                                headers={"User-Agent": "NoRegretsRVC", "Cache-Control": "no-cache"},
                            )
                            with urllib.request.urlopen(req_alt, timeout=10) as r_up2:
                                body_bytes = r_up2.read()
                        body_txt = body_bytes.decode("utf-8", errors="replace")
                        if marker_str in body_txt and len(body_bytes) > 64:
                            if gh_path.endswith(".py"):
                                compile(body_txt, gh_path, "exec")
                            candidates[gh_path] = (dst_path, marker_str, body_bytes)
                    except Exception as exc_u:
                        upd_errors.append(f"{gh_path}: {exc_u}")
                if not candidates:
                    for nb_rel in ("NoRegretsRVC_Kaggle.ipynb", "assets/NoRegretsRVC_Kaggle.ipynb", "public/NoRegretsRVC_Kaggle.ipynb"):
                        try:
                            req_nb = urllib.request.Request(
                                f"https://raw.githubusercontent.com/HunterThatsMe/NoRegretsRVC/main/{nb_rel}?t={ts_bust}",
                                headers={"User-Agent": "NoRegretsRVC", "Cache-Control": "no-cache"},
                            )
                            with urllib.request.urlopen(req_nb, timeout=14) as r_nb:
                                nb_obj = json.loads(r_nb.read().decode("utf-8", errors="replace"))
                            cell_src = chr(10).join(
                                "".join(c.get("source", [])) if isinstance(c.get("source"), list) else str(c.get("source", ""))
                                for c in nb_obj.get("cells", []) if c.get("cell_type") == "code"
                            )
                            for pat_rel, label_rel, dst_path, marker_str in (
                                (r'\\(repo_dir / "streamlit_app\\.py"\\)\\.write_bytes\\(base64\\.b64decode\\("([A-Za-z0-9+/=]+)"\\)\\)', "streamlit_app.py", Path(now_dir) / "streamlit_app.py", "CRIMSON_CSS"),
                                (r'\\(repo_dir / "noregrets_turbo\\.py"\\)\\.write_bytes\\(base64\\.b64decode\\("([A-Za-z0-9+/=]+)"\\)\\)', "noregrets_turbo.py", Path(now_dir) / "noregrets_turbo.py", "def apply("),
                                (r'\\(repo_dir / "\\.streamlit" / "config\\.toml"\\)\\.write_bytes\\(base64\\.b64decode\\("([A-Za-z0-9+/=]+)"\\)\\)', ".streamlit/config.toml", Path(now_dir) / ".streamlit" / "config.toml", "primaryColor"),
                            ):
                                bm = re.search(pat_rel, cell_src)
                                if bm:
                                    dec = _upd_b64.b64decode(bm.group(1))
                                    dec_txt = dec.decode("utf-8", errors="replace")
                                    if len(dec) > 64 and marker_str in dec_txt:
                                        if label_rel.endswith(".py"):
                                            compile(dec_txt, label_rel, "exec")
                                        candidates[label_rel] = (dst_path, marker_str, dec)
                            if candidates:
                                break
                        except Exception as exc_nb:
                            upd_errors.append(f"{nb_rel}: {exc_nb}")
                updated_files: list[str] = []
                skipped_older = False
                if candidates:
                    NR_STATE_DIR.mkdir(parents=True, exist_ok=True)
                    for label_rel, (dst_path, _, new_bytes) in candidates.items():
                        cur_bytes = dst_path.read_bytes() if dst_path.is_file() else b""
                        new_rev, _ = _extract_rev_and_ver(new_bytes)
                        cur_rev, _ = _extract_rev_and_ver(cur_bytes)
                        if cur_bytes and new_rev > 0 and cur_rev > 0 and new_rev < cur_rev:
                            skipped_older = True
                            continue
                        if cur_bytes == new_bytes:
                            continue
                        if cur_bytes:
                            try:
                                (NR_STATE_DIR / f"{dst_path.name}.bak").write_bytes(cur_bytes)
                            except Exception:
                                pass
                        dst_path.parent.mkdir(parents=True, exist_ok=True)
                        dst_path.write_bytes(new_bytes)
                        updated_files.append(label_rel)
                if updated_files:
                    _trigger_webui_restart(f"✔ Synced {', '.join(updated_files)} from GitHub main! Reloading WebUI in ~4 seconds (tunnel URL stays the same)...")
                elif candidates and not skipped_older:
                    st.session_state["nr_ver_status"] = f"✔ Already running the newest NoRegretsRVC build (Beta {APP_VERSION}) — no restart needed."
                elif skipped_older:
                    st.session_state["nr_ver_status"] = f"GitHub main currently hosts an older build revision than your installed Beta {APP_VERSION}. Kept your newer build intact."
                else:
                    st.session_state["nr_ver_status"] = f"Could not fetch valid WebUI files from GitHub main ({'; '.join(upd_errors) or 'repo raw unavailable'}). Your active build ({APP_VERSION}) was kept intact."
                st.rerun()
            if u3.button(_tr("Restart NoRegretsRVC"), type="secondary", use_container_width=True, key="nr_set_restart"):
                _trigger_webui_restart("Restarting NoRegretsRVC WebUI cleanly (~4 seconds — tunnel URL stays active)...")
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
            _step_head(3, "Kaggle Tunnel & Runtime Hardening", "Active Streamlit WebUI runtime, precision, Lossless Turbo mode, raw Dual T4 batch scaling, and CUDA session details.")
            st.session_state["nr_turbo_training"] = st.checkbox(
                "Lossless Turbo Training — Always ON by default (adaptive RAM cache, fused AdamW, cuDNN benchmark, DDP bucket view)",
                value=bool(st.session_state.get("nr_turbo_training", True)),
                key="nr_set_turbo_chk",
            )
            rt1, rt2 = st.columns(2, gap="small")
            rt1.markdown(f'<div class="nr-metric-box"><div class="nr-metric-k">Working Directory</div><div class="nr-metric-v">{html.escape(now_dir)}</div></div>', unsafe_allow_html=True)
            rt2.markdown(f'<div class="nr-metric-box"><div class="nr-metric-k">Active Precision</div><div class="nr-metric-v">{html.escape(st.session_state["nr_precision"].upper())}</div></div>', unsafe_allow_html=True)
            rt3, rt4 = st.columns(2, gap="small")
            rt3.markdown(f'<div class="nr-metric-box"><div class="nr-metric-k">GPU Allocation</div><div class="nr-metric-v">{html.escape(_gpu_info_lines(st.session_state["nr_gpu_num"]).replace(chr(10), " · "))}</div></div>', unsafe_allow_html=True)
            rt4.markdown('<div class="nr-metric-box"><div class="nr-metric-k">Dual T4 Batch Scaling</div><div class="nr-metric-v">Raw 2x per-GPU batch</div></div>', unsafe_allow_html=True)

# __NR_FOOTER_START__
_render_active_tab(active_tab)


def _render_bg_watcher_inner() -> None:
    _, done_w = _poll_core_tasks()
    _render_toasts()
    if done_w and _HAS_FRAGMENT:
        try:
            st.rerun(scope="app")
        except TypeError:
            st.rerun()


if _HAS_FRAGMENT:
    _bg_interval = 1.5 if (_has_active_bg_work() or bool(st.session_state.get("nr_toasts"))) else None

    @st.fragment(run_every=_bg_interval)
    def _nr_bg_watcher() -> None:
        _render_bg_watcher_inner()

    _nr_bg_watcher()
else:
    _render_bg_watcher_inner()
`;
}
