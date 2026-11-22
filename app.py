import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path


TITLE = "Hallucination Detection in Small Language Models for Science QA With and Without RAG"
SUBTITLE = "Can Small Language Models Tell When They Are Wrong?"
AUTHOR = "Sejal Singh"
EXTRA = "7th Semester Research Project | Babasaheb Bhimrao Ambedkar University | 2023-2027"   # <- guide / college / year yahan likho


AMBER, TEAL, INK, SLATE = "#e09f3e", "#1b998b", "#14213d", "#6b7a8f"

st.set_page_config(page_title="Hallucination Detection", page_icon="🔍", layout="wide")
st.markdown(f"""<style>
@import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:wght@600;700&display=swap');
h1,h2,h3 {{ font-family:'Source Serif 4',Georgia,serif !important; }}
.hero {{ background:{INK}; color:#fff; padding:2.6rem 2.2rem; border-radius:6px; margin-bottom:1.4rem;
        border-bottom:6px solid {AMBER}; }}
.hero h1 {{ color:#fff; font-size:2.5rem; line-height:1.2; margin:0 0 .6rem; }}
.hero p {{ color:#dfe3ec; margin:.2rem 0; font-size:1.05rem; }}
.hero .by {{ margin-top:1.1rem; color:#fff; font-weight:600; }}
.card {{ padding:1rem 1.2rem; border:1px solid rgba(128,128,128,.3); border-radius:6px;
        background:rgba(128,128,128,.06); height:100%; }}
.d0 {{ border-left:6px solid {AMBER}; }}
.d1 {{ border-left:6px solid {TEAL}; }}
.card h4 {{ margin:0 0 .4rem; }}
</style>""", unsafe_allow_html=True)


@st.cache_data
def load():
    return pd.read_csv(Path(__file__).parent / "data" / "summary.csv")


df = load()
df["Model"] = df.model.map({"qwen3_4b": "Qwen3-4B", "phi4_mini": "Phi-4-mini"})
df["Setting"] = df.condition.map({"no_ctx": "Without context", "ctx": "With context"})

# ---------- Hero ----------
st.markdown(f"""<div class="hero"><h1>{TITLE}</h1><p>{SUBTITLE}</p>
<p class="by">Research by {AUTHOR}</p><p>{EXTRA}</p></div>""", unsafe_allow_html=True)

# ---------- Introduction ----------
st.header("Introduction")
st.write(
    "Language models sometimes give a wrong answer in a confident voice. This is called a hallucination. "
    "If we could detect such answers without knowing the right answer, we could warn the user. "
    "This project tests two simple detectors on two small open models. We ask the same question "
    "several times and check how much the model contradicts itself. We run the test twice: "
    "once with the question alone, and once with passages retrieved from a text collection added to the prompt."
)

# ---------- Terms ----------
st.header("What was used")
c1, c2 = st.columns(2)
c1.markdown("""<div class="card"><h4>Models</h4>
<b>Qwen3-4B</b> (Qwen3-4B-Instruct-2507) and <b>Phi-4-mini</b> (Phi-4-mini-instruct).
Both run in half precision on a single Colab T4 GPU.</div>""", unsafe_allow_html=True)
c2.markdown("""<div class="card"><h4>Data and setup</h4>
ARC-Challenge science questions: 150 for development and 300 for testing (fixed seed 42).
For each question the model gives one main answer (greedy), then 5 extra answers sampled at temperature 0.7.
With context, retrieved passages are added to the prompt.</div>""", unsafe_allow_html=True)

st.write("")
c3, c4, c5 = st.columns(3)
c3.markdown("""<div class="card d0"><h4>D0: answer agreement</h4>
Score = 1 minus the share of the 5 sampled answers that match the main answer.
If the model keeps changing its answer, D0 is high and the answer is likely wrong.</div>""", unsafe_allow_html=True)
c4.markdown("""<div class="card d1"><h4>D1: explanation similarity</h4>
Each answer comes with a one-sentence reason. D1 is 1 minus the average embedding similarity
between the main reason and the 5 sampled reasons, using all-MiniLM-L6-v2.
It can still react when all 5 letters agree.</div>""", unsafe_allow_html=True)
c5.markdown("""<div class="card"><h4>AUROC</h4>
How well a score separates wrong answers from correct ones.
0.5 means no better than chance and 1.0 is perfect.
Error bars show 95% bootstrap confidence intervals.</div>""", unsafe_allow_html=True)

# ---------- Results ----------
st.header("Results")
split = st.radio("Question set", ["test", "dev"], horizontal=True,
                 format_func=lambda s: "Test set (300 questions)" if s == "test" else "Dev set (150 questions)")
d = df[df.split == split]
tab1, tab2, tab3, tab4 = st.tabs(["Accuracy", "Detection quality", "Effect of context", "Full table"])

with tab1:
    st.subheader("Does retrieved context help the model answer correctly?")
    fig = px.bar(d, x="Model", y="accuracy", color="Setting", barmode="group", text_auto=".3f",
                 color_discrete_map={"Without context": SLATE, "With context": INK})
    fig.update_yaxes(range=[0.5, 1], title="Accuracy")
    st.plotly_chart(fig, width='stretch')
    st.caption("Context did not raise accuracy for either model. The differences are only a few questions.")

with tab2:
    st.subheader("How well do D0 and D1 spot wrong answers?")
    rows = []
    for _, r in d.iterrows():
        for k, lab in (("d0", "D0 answer agreement"), ("d1", "D1 explanation similarity")):
            rows.append(dict(Group=f"{r.Model}, {r.Setting.lower()}", Detector=lab, AUROC=r[f"{k}_auroc"],
                             up=r[f"{k}_hi"] - r[f"{k}_auroc"], down=r[f"{k}_auroc"] - r[f"{k}_lo"]))
    a = pd.DataFrame(rows)
    fig = px.bar(a, x="Group", y="AUROC", color="Detector", barmode="group", error_y="up", error_y_minus="down",
                 color_discrete_map={"D0 answer agreement": AMBER, "D1 explanation similarity": TEAL})
    fig.add_hline(y=0.5, line_dash="dash", annotation_text="chance (0.5)")
    fig.update_yaxes(range=[0.3, 1])
    st.plotly_chart(fig, width='stretch')
    st.caption("Bars whose error bars reach the dashed line are not clearly better than guessing.")

with tab3:
    st.subheader("Paired differences on the test set (AUROC)")
    pd_rows = [("Qwen3-4B, without context: D1 minus D0", .065, -.046, .174),
               ("Qwen3-4B, with context: D1 minus D0", .067, -.032, .158),
               ("Phi-4-mini, without context: D1 minus D0", .013, -.075, .096),
               ("Phi-4-mini, with context: D1 minus D0", -.072, -.154, .011),
               ("Qwen3-4B D0: context minus no context", -.016, -.073, .034),
               ("Qwen3-4B D1: context minus no context", -.014, -.157, .125),
               ("Phi-4-mini D0: context minus no context", .059, -.038, .152),
               ("Phi-4-mini D1: context minus no context", -.026, -.109, .054)]
    p = pd.DataFrame(pd_rows, columns=["Comparison", "Difference", "Low", "High"])
    p["Clear?"] = (p.Low > 0) | (p.High < 0)
    fig = px.scatter(p, x="Difference", y="Comparison", error_x=p.High - p.Difference,
                     error_x_minus=p.Difference - p.Low, color_discrete_sequence=[INK])
    fig.add_vline(x=0, line_dash="dash")
    fig.update_yaxes(autorange="reversed", title="")
    st.plotly_chart(fig, width='stretch')
    st.dataframe(p, hide_index=True, width='stretch')
    st.caption("Every interval includes 0, so none of these differences is statistically clear with this sample size.")

with tab4:
    show = d[["Model", "Setting", "n", "errors", "accuracy", "d0_auroc", "d0_lo", "d0_hi", "d1_auroc", "d1_lo", "d1_hi"]]
    st.dataframe(show, hide_index=True, width='stretch')

# ---------- Conclusion ----------
st.header("Conclusion")
st.markdown("""
- **Context did not help.** Adding retrieved passages left accuracy about the same (Qwen3-4B 0.893 to 0.880, Phi-4-mini 0.820 to 0.797 on the test set).
- **Qwen3-4B is confidently wrong.** In 99.7% of test questions all 5 samples gave the same answer, so D0 is close to chance (0.51 to 0.53).
- **Phi-4-mini gives a usable but modest signal.** Its D0 and D1 AUROC lie roughly between 0.63 and 0.70 on the test set.
- **No detector was clearly better than the other**, and context did not clearly change detection quality.
- **Limitation:** there are only 32 to 61 wrong answers per setting, so the confidence intervals are wide. A larger test set is the next step.
""")
st.divider()
st.caption(f"{AUTHOR} · {EXTRA}")
