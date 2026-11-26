"""Interactive Streamlit replay, dictionary, observatory, and human arena UI."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from babel.ui.components.stage_svg import COLOR_NAMES, SHAPE_NAMES, render_stage_html
from babel.ui.loader import list_run_dirs, load_run_bundle, load_summaries, predict_with_saved_listener

RUNS_DIR = Path("runs")


def _extract_eval_arrays(bundle: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
    """Return (messages, preds, truth, is_synthetic_world) from saved predictions.npz."""
    preds_map = bundle["predictions"]
    if "messages" in preds_map and "truth" in preds_map:
        msgs = preds_map["messages"]
        truth = preds_map["truth"]
        # Infer listener predictions directly from saved listener checkpoint
        preds = []
        for m in msgs:
            probs = predict_with_saved_listener(bundle["run_dir"], m)
            preds.append([int(np.argmax(p)) for p in probs])
        return msgs, np.asarray(preds, dtype=np.int64), truth, True

    split_prefix = "test" if "test_messages" in preds_map else "train"
    msgs = preds_map[f"{split_prefix}_messages"]
    preds = preds_map[f"{split_prefix}_preds"]
    truth = preds_map[f"{split_prefix}_truth"]
    if truth.ndim == 1:
        truth = truth[:, np.newaxis]
    if preds.ndim == 1:
        preds = preds[:, np.newaxis]
    is_synth = bundle["config"]["config"].get("dataset") == "synthetic" or truth.shape[1] == 2 and int(truth.max()) <= 2 and len(truth) <= 9
    return msgs, preds, truth, bool(is_synth)


def _format_label(vec: np.ndarray, is_synth: bool) -> str:
    if is_synth and len(vec) == 2:
        return f"{SHAPE_NAMES[int(vec[0])]} / {COLOR_NAMES[int(vec[1])]}"
    return ", ".join(str(int(v)) for v in vec)


def main() -> None:
    st.set_page_config(page_title="Babel — Emergent Symbolic Communication", layout="wide")
    st.title("Babel: Emergent Symbolic Communication Observatory")
    st.caption("Powered strictly by saved run artifacts under runs/ — zero training or hardcoded metrics during demo.")

    run_dirs = list_run_dirs(RUNS_DIR)
    summaries = load_summaries(RUNS_DIR)
    if not run_dirs and not summaries:
        st.warning("No saved runs found under `runs/`. Run `make reproduce` or `babel train` first.")
        return

    tabs = st.tabs(["Replay Theater", "Live Dictionary", "Observatory", "Human Arena"])

    # ------------------------------------------------------------------
    # Tab 1: Replay Theater
    # ------------------------------------------------------------------
    with tabs[0]:
        if not run_dirs:
            st.info("No individual runs with predictions.npz found yet.")
        else:
            selected_name = st.selectbox("Select Saved Run", [d.name for d in run_dirs], key="replay_run")
            bundle = load_run_bundle(RUNS_DIR / selected_name)
            msgs, preds, truth, is_synth = _extract_eval_arrays(bundle)
            n_items = len(msgs)
            col_a, col_b, col_c = st.columns([2, 1, 1])
            with col_a:
                idx = st.slider("Sample / Object Scrubber", 0, max(0, n_items - 1), 0, key="replay_idx")
            with col_b:
                speed = st.select_slider("Autoplay Delay (s)", options=[0.25, 0.5, 1.0, 2.0], value=0.75)
            with col_c:
                autoplay = st.button("Play Sequence")

            placeholder = st.empty()

            def _draw_step(step_i: int) -> None:
                m = [int(v) for v in msgs[step_i]]
                t_vec = truth[step_i]
                p_vec = preds[step_i]
                ok = bool(np.array_equal(t_vec, p_vec))
                html = render_stage_html(
                    message=m,
                    is_correct=ok,
                    true_label_text=_format_label(t_vec, is_synth),
                    pred_label_text=_format_label(p_vec, is_synth),
                    synthetic_truth=(int(t_vec[0]), int(t_vec[1])) if is_synth and len(t_vec) == 2 else None,
                    synthetic_pred=(int(p_vec[0]), int(p_vec[1])) if is_synth and len(p_vec) == 2 else None,
                    reward_value=1.0 if ok else 0.0,
                )
                placeholder.markdown(html, unsafe_allow_html=True)

            if autoplay:
                for i in range(min(n_items, 20)):
                    _draw_step(i)
                    time.sleep(float(speed))
            else:
                _draw_step(idx)

            if bundle["metrics"]:
                df_m = pd.DataFrame(bundle["metrics"])
                x_col = "episode" if "episode" in df_m.columns else "epoch"
                y_cols = [c for c in ("exact_accuracy", "train_accuracy", "val_accuracy", "reward") if c in df_m.columns]
                if y_cols and x_col in df_m.columns:
                    st.plotly_chart(px.line(df_m, x=x_col, y=y_cols, title="Training Trajectory (from metrics.jsonl)"), use_container_width=True)

    # ------------------------------------------------------------------
    # Tab 2: Live Dictionary
    # ------------------------------------------------------------------
    with tabs[1]:
        if not run_dirs:
            st.info("No saved runs available.")
        else:
            dict_name = st.selectbox("Select Run for Lexicon", [d.name for d in run_dirs], key="dict_run")
            bundle = load_run_bundle(RUNS_DIR / dict_name)
            msgs, preds, truth, is_synth = _extract_eval_arrays(bundle)
            vocab_k = int(max(msgs.max(), 1) + 1)
            msg_len = msgs.shape[1]

            rows = []
            for i in range(len(msgs)):
                rows.append({
                    "Index": i,
                    "Meaning": _format_label(truth[i], is_synth),
                    "Message": " ".join(f"m{p}={int(msgs[i, p])}" for p in range(msg_len)),
                    "Listener Pred": _format_label(preds[i], is_synth),
                    "Correct": bool(np.array_equal(truth[i], preds[i])),
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True)

            if is_synth and msg_len == 2 and truth.shape[1] == 2:
                st.subheader("Positional Symbol × Attribute Heatmaps (L = 2)")
                c1, c2 = st.columns(2)
                for pos, col_ui in enumerate((c1, c2)):
                    mat = np.zeros((vocab_k, len(SHAPE_NAMES) + len(COLOR_NAMES)), dtype=int)
                    for i in range(len(msgs)):
                        sym = int(msgs[i, pos])
                        mat[sym, int(truth[i, 0])] += 1
                        mat[sym, len(SHAPE_NAMES) + int(truth[i, 1])] += 1
                    x_labels = list(SHAPE_NAMES) + list(COLOR_NAMES)
                    y_labels = [f"Sym {s}" for s in range(vocab_k)]
                    fig = px.imshow(mat, x=x_labels, y=y_labels, text_auto=True, color_continuous_scale="Blues",
                                    title=f"Position {pos + 1} Symbol Usage")
                    col_ui.plotly_chart(fig, use_container_width=True)
            else:
                mat = np.zeros((vocab_k, int(truth[:, 0].max() + 1)), dtype=int)
                for i in range(len(msgs)):
                    mat[int(msgs[i, 0]), int(truth[i, 0])] += 1
                fig = px.imshow(
                    mat,
                    x=[f"Class {c}" for c in range(mat.shape[1])],
                    y=[f"Sym {s}" for s in range(vocab_k)],
                    text_auto=True,
                    color_continuous_scale="Teal",
                    title="Position 1 Symbol × Target Class Count",
                )
                st.plotly_chart(fig, use_container_width=True)

    # ------------------------------------------------------------------
    # Tab 3: Observatory
    # ------------------------------------------------------------------
    with tabs[2]:
        if not summaries:
            st.info("No saved summaries in `runs/summaries/`. Run `babel reproduce` to generate them.")
        else:
            for exp_name, payload in summaries.items():
                kind = payload.get("kind")
                if kind == "rate_accuracy":
                    st.subheader(f"Rate-Accuracy Curve: {exp_name} ({payload['dataset']})")
                    records = []
                    for k_str, row in sorted(payload["results_by_vocab"].items(), key=lambda x: int(x[0])):
                        bits = row["bits_capacity"]
                        records.append({
                            "Bits": bits, "K": int(k_str), "Method": "Emergent Neural",
                            "Accuracy": row["neural"]["accuracy"]["mean"],
                            "CI95": row["neural"]["accuracy"]["ci95"],
                        })
                        for bname, label in (
                            ("kmeans_vq", "k-Means VQ"),
                            ("size_limited_tree", "Size-Limited Tree"),
                            ("full_data_oracle", "Oracle (LogReg)"),
                            ("full_data_oracle_mlp", "Oracle (MLP)"),
                        ):
                            records.append({
                                "Bits": bits, "K": int(k_str), "Method": label,
                                "Accuracy": row["baselines"][bname]["accuracy"]["mean"],
                                "CI95": row["baselines"][bname]["accuracy"]["ci95"],
                            })
                    df_ra = pd.DataFrame(records)
                    fig = px.line(df_ra, x="Bits", y="Accuracy", color="Method", error_y="CI95", markers=True,
                                  title=f"{exp_name}: Test Accuracy vs Bit Budget (log2 K)")
                    st.plotly_chart(fig, use_container_width=True)

                elif kind == "grammar":
                    st.subheader(f"Grammar & Held-Out Generalization: {exp_name} ({payload['dataset']})")
                    g_rows = []
                    for mode, by_k in payload["results_by_mode"].items():
                        for k_str, row in sorted(by_k.items(), key=lambda x: int(x[0])):
                            m = row["metrics"]
                            g_rows.append({
                                "Mode": mode, "K": int(k_str),
                                "PosDis": m["posdis"]["mean"],
                                "TopSim": m["topsim"]["mean"],
                                "SeenAcc": m["seen_exact_accuracy"]["mean"],
                                "HeldOutExact": m["heldout_exact_accuracy"]["mean"],
                                "HeldOutAttr": m["heldout_attr_accuracy"]["mean"],
                            })
                    df_g = pd.DataFrame(g_rows)
                    c1, c2 = st.columns(2)
                    c1.plotly_chart(px.line(df_g, x="K", y="PosDis", color="Mode", markers=True, title="PosDis vs K"), use_container_width=True)
                    c2.plotly_chart(px.line(df_g, x="K", y="HeldOutAttr", color="Mode", markers=True, title="Held-Out Zero-Shot Attr Acc vs K"), use_container_width=True)

                elif kind == "dialect_matrix":
                    st.subheader(f"Dialect Cross-Play Matrix: {exp_name}")
                    for ds_name, drow in payload["datasets"].items():
                        st.markdown(f"**{ds_name}** — Diagonal: `{drow['diagonal_mean']:.3f}` | Raw Off-Diag: `{drow['raw_off_diagonal_mean']:.3f}` | Hungarian-Aligned Off-Diag: `{drow['hungarian_off_diagonal_mean']:.3f}`")
                        c1, c2 = st.columns(2)
                        c1.plotly_chart(px.imshow(drow["raw_matrix"], text_auto=".2f", zmin=0, zmax=1, title=f"{ds_name} Raw A_ij"), use_container_width=True)
                        c2.plotly_chart(px.imshow(drow["aligned_matrix"], text_auto=".2f", zmin=0, zmax=1, title=f"{ds_name} Hungarian-Aligned A_ij"), use_container_width=True)

    # ------------------------------------------------------------------
    # Tab 4: Human Arena
    # ------------------------------------------------------------------
    with tabs[3]:
        compact_runs = []
        for d in run_dirs:
            b = load_run_bundle(d)
            k_val = b["config"]["config"].get("vocab_size", 9)
            if int(k_val) <= 6 and (b["has_npz_ckpt"] or b["has_pt_ckpt"]):
                compact_runs.append(d)
        if not compact_runs:
            st.info("No compact-vocabulary runs (K <= 6) with saved checkpoints found.")
        else:
            arena_name = st.selectbox("Select Compact AI Partner (K <= 6)", [d.name for d in compact_runs], key="arena_run")
            bundle = load_run_bundle(RUNS_DIR / arena_name)
            msgs, preds, truth, is_synth = _extract_eval_arrays(bundle)
            vocab_k = int(bundle["config"]["config"].get("vocab_size", max(msgs.max(), 1) + 1))
            msg_len = msgs.shape[1]

            if "arena_idx" not in st.session_state:
                st.session_state.arena_idx = 0
                st.session_state.arena_hits = 0
                st.session_state.arena_trials = 0

            if st.button("New Random Round"):
                st.session_state.arena_idx = int(np.random.default_rng().integers(len(msgs)))

            cur_i = st.session_state.arena_idx % len(msgs)
            mode_choice = st.radio("Arena Role", ["Play as Listener (Decode AI Symbols)", "Play as Speaker (Steer AI Listener)"], horizontal=True)

            if mode_choice.startswith("Play as Listener"):
                ai_msg = [int(v) for v in msgs[cur_i]]
                st.markdown(f"### AI Speaker emitted: `{ai_msg}`")
                if is_synth and truth.shape[1] == 2:
                    g_shape = st.selectbox("Guess Shape", list(range(len(SHAPE_NAMES))), format_func=lambda i: SHAPE_NAMES[i])
                    g_color = st.selectbox("Guess Color", list(range(len(COLOR_NAMES))), format_func=lambda i: COLOR_NAMES[i])
                    if st.button("Submit Listener Guess"):
                        ok = (g_shape == int(truth[cur_i, 0])) and (g_color == int(truth[cur_i, 1]))
                        st.session_state.arena_trials += 1
                        st.session_state.arena_hits += int(ok)
                        st.success("Correct!" if ok else f"Incorrect — true object was {_format_label(truth[cur_i], True)}")
                else:
                    n_cls = int(truth[:, 0].max() + 1)
                    g_cls = st.selectbox("Guess Class ID", list(range(n_cls)))
                    if st.button("Submit Listener Guess"):
                        ok = g_cls == int(truth[cur_i, 0])
                        st.session_state.arena_trials += 1
                        st.session_state.arena_hits += int(ok)
                        st.success("Correct!" if ok else f"Incorrect — true class was {int(truth[cur_i, 0])}")
            else:
                st.markdown(f"### Target to communicate: **{_format_label(truth[cur_i], is_synth)}**")
                chosen = [
                    st.selectbox(f"Position {p + 1} Symbol", list(range(vocab_k)), key=f"hum_sym_{p}")
                    for p in range(msg_len)
                ]
                if st.button("Send Message to AI Listener"):
                    probs = predict_with_saved_listener(bundle["run_dir"], chosen)
                    ai_pred = np.array([int(np.argmax(p)) for p in probs], dtype=np.int64)
                    ok = bool(np.array_equal(ai_pred, truth[cur_i]))
                    st.session_state.arena_trials += 1
                    st.session_state.arena_hits += int(ok)
                    st.markdown(
                        render_stage_html(
                            message=chosen,
                            is_correct=ok,
                            true_label_text=_format_label(truth[cur_i], is_synth),
                            pred_label_text=_format_label(ai_pred, is_synth),
                            synthetic_truth=(int(truth[cur_i, 0]), int(truth[cur_i, 1])) if is_synth and truth.shape[1] == 2 else None,
                            synthetic_pred=(int(ai_pred[0]), int(ai_pred[1])) if is_synth and len(ai_pred) == 2 else None,
                        ),
                        unsafe_allow_html=True,
                    )

            st.caption(f"Human Arena Score: {st.session_state.arena_hits} / {st.session_state.arena_trials}")


if __name__ == "__main__":
    main()
