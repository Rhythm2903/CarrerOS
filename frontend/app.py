import requests
import streamlit as st

API_BASE = "http://localhost:8000"

st.set_page_config(
    page_title="AI Career Copilot",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Space Grotesk', sans-serif; }
.stApp { background: linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 100%); }
.metric-card {
    background: linear-gradient(135deg, #1e1e3a, #252545);
    border: 1px solid #3a3a6a; border-radius: 12px;
    padding: 20px; text-align: center; margin: 8px 0;
}
.score-ring {
    font-size: 3rem; font-weight: 700;
    background: linear-gradient(135deg, #6c63ff, #a78bfa);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.skill-chip { display:inline-block;padding:4px 12px;border-radius:20px;margin:3px;font-size:0.8rem;font-weight:600; }
.skill-match  { background:#1a3a2a;color:#4ade80;border:1px solid #4ade80; }
.skill-missing{ background:#3a1a1a;color:#f87171;border:1px solid #f87171; }
.skill-tier1  { background:#3a2a1a;color:#fb923c;border:1px solid #fb923c; }
.skill-tier2  { background:#1a2a3a;color:#60a5fa;border:1px solid #60a5fa; }
.agent-step   { background:#1a1a2e;border-left:3px solid #6c63ff;padding:8px 12px;margin:4px 0;border-radius:0 8px 8px 0;font-size:0.8rem; }
.market-card  { background:#1a2a1a;border:1px solid #4ade80;border-radius:12px;padding:16px;margin:8px 0; }
.token-badge  { background:#252545;border-radius:8px;padding:6px 12px;font-size:0.75rem;color:#a78bfa;display:inline-block;margin:4px; }
.chat-user    { background:#252545;border-radius:12px;padding:12px 16px;margin:6px 0; }
.chat-bot     { background:#1a2a1a;border-radius:12px;padding:12px 16px;margin:6px 0;border-left:3px solid #4ade80; }
.stButton>button { background:linear-gradient(135deg,#6c63ff,#a78bfa);color:white;border:none;border-radius:8px;font-weight:600;padding:10px 24px; }
</style>
""",
    unsafe_allow_html=True,
)

for key in [
    "session_id",
    "resume_data",
    "analysis",
    "chat_history",
    "uploaded",
    "uploaded_file_name",
]:
    if key not in st.session_state:
        st.session_state[key] = None if key != "chat_history" else []

with st.sidebar:
    st.markdown("## 🚀 AI Career Copilot")
    st.markdown("*Agentic Multi-Model Career Intelligence*")
    st.divider()
    uploaded_file = st.file_uploader("📄 Upload Resume (PDF)", type=["pdf"])

    should_upload = uploaded_file and (
        not st.session_state.uploaded
        or uploaded_file.name != st.session_state.uploaded_file_name
    )

    if should_upload:
        with st.spinner("Parsing & indexing resume..."):
            if st.session_state.session_id:
                requests.delete(f"{API_BASE}/session/{st.session_state.session_id}")
            resp = requests.post(
                f"{API_BASE}/upload_resume",
                files={
                    "file": (
                        uploaded_file.name,
                        uploaded_file.getvalue(),
                        "application/pdf",
                    )
                },
            )
            if resp.status_code == 200:
                data = resp.json()
                st.session_state.session_id = data["session_id"]
                st.session_state.resume_data = data["resume_data"]
                st.session_state.analysis = None
                st.session_state.chat_history = []
                st.session_state.uploaded = True
                st.session_state.uploaded_file_name = uploaded_file.name
                st.success(f"✅ Indexed! ({data['chunks_indexed']} chunks)")
            else:
                st.error(f"Upload failed: {resp.text}")

    if st.session_state.uploaded:
        rd = st.session_state.resume_data
        st.markdown("**📋 Resume Loaded**")
        if rd.get("name"):
            st.markdown(f"👤 **{rd['name']}**")
        if rd.get("email"):
            st.markdown(f"📧 {rd['email']}")
        if rd.get("skills"):
            st.markdown(f"🔧 {len(rd['skills'])} skills detected")
        st.divider()
        if st.button("🗑️ Clear Session"):
            requests.delete(f"{API_BASE}/session/{st.session_state.session_id}")
            for key in [
                "session_id",
                "resume_data",
                "analysis",
                "uploaded",
                "uploaded_file_name",
            ]:
                st.session_state[key] = None
            st.session_state.chat_history = []
            st.rerun()

st.markdown("# 🚀 AI Career Copilot")
st.markdown("*7 specialized AI agents • OpenAI-powered • Market-aware • Self-critique loop*")

if not st.session_state.uploaded:
    st.info("👈 Upload your PDF resume in the sidebar to get started.")
    st.stop()

tab1, tab2, tab3, tab4 = st.tabs(
    ["📊 Match Analysis", "🗺️ Learning Roadmap", "📈 Market Intelligence", "💬 Chat"]
)

with tab1:
    st.markdown("### Paste a Job Description")
    job_desc = st.text_area(
        "JD",
        height=200,
        placeholder="Paste full job description here...",
        label_visibility="collapsed",
    )

    if st.button("🤖 Run Agentic Analysis", use_container_width=True):
        if not job_desc.strip():
            st.warning("Please paste a job description first.")
        else:
            with st.spinner("7 agents working in sequence... (20-35 seconds)"):
                resp = requests.post(
                    f"{API_BASE}/analyze_job",
                    json={
                        "session_id": st.session_state.session_id,
                        "job_description": job_desc,
                    },
                )
                if resp.status_code == 200:
                    st.session_state.analysis = resp.json()
                else:
                    st.error(f"Analysis failed: {resp.text}")

    if st.session_state.analysis:
        analysis = st.session_state.analysis
        ma = analysis.get("match_analysis", {})
        sg = analysis.get("skill_gap", {})
        meta = analysis.get("meta", {})

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(
                f'<div class="metric-card"><div style="color:#a78bfa;font-size:0.85rem">MATCH SCORE</div><div class="score-ring">{ma.get("match_score", "?")}%</div></div>',
                unsafe_allow_html=True,
            )
        with col2:
            signal_color = {
                "strong_yes": "#4ade80",
                "yes": "#86efac",
                "maybe": "#fbbf24",
                "no": "#f87171",
            }.get(ma.get("hiring_signal", "maybe"), "#a78bfa")
            st.markdown(
                f'<div class="metric-card"><div style="color:#a78bfa;font-size:0.85rem">HIRING SIGNAL</div><div style="font-size:1.4rem;font-weight:700;color:{signal_color}">{ma.get("hiring_signal", "?").upper().replace("_", " ")}</div></div>',
                unsafe_allow_html=True,
            )
        with col3:
            severity_color = {
                "low": "#4ade80",
                "medium": "#fbbf24",
                "high": "#fb923c",
                "critical": "#f87171",
            }.get(sg.get("gap_severity", "medium"), "#a78bfa")
            st.markdown(
                f'<div class="metric-card"><div style="color:#a78bfa;font-size:0.85rem">GAP SEVERITY</div><div style="font-size:1.4rem;font-weight:700;color:{severity_color}">{sg.get("gap_severity", "?").upper()}</div></div>',
                unsafe_allow_html=True,
            )
        with col4:
            st.markdown(
                f'<div class="metric-card"><div style="color:#a78bfa;font-size:0.85rem">TOKEN EFFICIENCY</div><div style="font-size:0.9rem;color:#e2e2ff;margin-top:8px">{meta.get("efficiency", "")}</div></div>',
                unsafe_allow_html=True,
            )

        st.markdown(f"*{ma.get('one_line_verdict', '')}*")
        st.divider()

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**✅ Matched Skills**")
            chips = " ".join(
                [
                    f'<span class="skill-chip skill-match">{skill}</span>'
                    for skill in ma.get("matched_skills", [])
                ]
            )
            st.markdown(chips or "*None found*", unsafe_allow_html=True)
        with col_b:
            st.markdown("**❌ Missing Critical**")
            chips = " ".join(
                [
                    f'<span class="skill-chip skill-missing">{skill}</span>'
                    for skill in ma.get("missing_critical", [])
                ]
            )
            st.markdown(chips or "*None*", unsafe_allow_html=True)

        st.divider()
        col_s, col_c = st.columns(2)
        with col_s:
            st.markdown("**Strengths For This Role**")
            for item in ma.get("strengths_for_role", []):
                st.markdown(f"- {item}")
            for item in sg.get("resume_strength_signals", []):
                st.markdown(f"- {item}")
        with col_c:
            st.markdown("**Main Concerns**")
            for item in ma.get("concerns", []):
                st.markdown(f"- {item}")
            if sg.get("experience_gap"):
                st.markdown(f"- {sg.get('experience_gap')}")

        st.divider()
        st.markdown("### 🔬 Skill Gap Tiers")
        if sg.get("tier1_blockers"):
            st.markdown("**🔴 Tier 1 - Blockers (fix first)**")
            for gap in sg["tier1_blockers"]:
                st.markdown(
                    f'<span class="skill-chip skill-tier1">{gap["skill"]}</span> - {gap.get("why_critical", "")} *(~{gap.get("estimated_learn_time", "?")})*',
                    unsafe_allow_html=True,
                )
        if sg.get("tier2_differentiators"):
            st.markdown("**🔵 Tier 2 - Differentiators (stand out)**")
            for gap in sg["tier2_differentiators"]:
                st.markdown(
                    f'<span class="skill-chip skill-tier2">{gap["skill"]}</span> - {gap.get("impact", "")}',
                    unsafe_allow_html=True,
                )
        if sg.get("biggest_strength_to_highlight"):
            st.markdown(f"**Best Story To Highlight:** {sg.get('biggest_strength_to_highlight')}")
        if sg.get("gap_severity_reason"):
            st.markdown(f"**Why This Severity:** {sg.get('gap_severity_reason')}")

        with st.expander("🔍 Agent Execution Trace"):
            icons = {
                "parser": "🔍",
                "jd_decomposer": "📋",
                "matcher": "⚖️",
                "gap_analyzer": "🔬",
                "market_intel": "📈",
                "roadmap_builder": "🗺️",
                "critic": "🧐",
                "orchestrator": "🎯",
            }
            for step in meta.get("agent_trace", []):
                icon = icons.get(step["agent"], "🤖")
                detail = f"· {step['detail']}" if step.get("detail") else ""
                st.markdown(
                    f'<div class="agent-step">{icon} <b>{step["agent"]}</b> -> {step["action"]} {detail}</div>',
                    unsafe_allow_html=True,
                )

            critique = meta.get("critique", {})
            if critique:
                approved_color = "#4ade80" if critique.get("approved") else "#f87171"
                badge_html = (
                    f'<div class="token-badge">Quality Score: {critique.get("quality_score")}/10</div> '
                    f'<div class="token-badge" style="color:{approved_color}">{"✅ Approved" if critique.get("approved") else "⚠️ Flagged"}</div>'
                )
                st.markdown(badge_html, unsafe_allow_html=True)

with tab2:
    if not st.session_state.analysis:
        st.info("📊 Run analysis first (Tab 1).")
    else:
        rm = st.session_state.analysis.get("learning_roadmap", {})
        st.markdown(f"## 🗺️ {rm.get('title', 'Your Learning Roadmap')}")
        st.markdown(f"*{rm.get('executive_summary', '')}*")

        st.markdown("### ⚡ Quick Wins - Do This Week")
        for item in rm.get("quick_wins_week1", []):
            st.markdown(f"- 🎯 {item}")

        st.markdown("### 📅 90-Day Plan")
        for phase in rm.get("phases", []):
            with st.expander(
                f"**Phase {phase['phase']}: {phase['title']}** - {phase.get('days', '')}",
                expanded=phase["phase"] == 1,
            ):
                st.markdown(f"**Goal:** {phase.get('primary_goal', '')}")
                st.markdown("**Tasks:**")
                for task in phase.get("tasks", []):
                    st.markdown(
                        f"- [{task.get('hours_per_week', '?')}h/wk] **{task['task']}** - *{task.get('resource', '')}*"
                    )
                st.markdown(f"**✅ Milestone:** {phase.get('milestone', '')}")

        st.markdown("### 🏗️ Portfolio Projects")
        for project in rm.get("portfolio_projects", []):
            st.markdown(f"**{project['name']}** - {project['description']}")
            chips = " ".join(
                [
                    f'<span class="skill-chip skill-tier2">{skill}</span>'
                    for skill in project.get("skills_demonstrated", [])
                ]
            )
            st.markdown(chips, unsafe_allow_html=True)

        st.markdown("### 🎯 Interview Prep Focus")
        for item in rm.get("interview_prep_focus", []):
            st.markdown(f"- {item}")

with tab3:
    if not st.session_state.analysis:
        st.info("📊 Run analysis first (Tab 1).")
    else:
        mi = st.session_state.analysis.get("market_intel", {})
        st.markdown("### 📈 Market Intelligence")

        col1, col2 = st.columns(2)
        with col1:
            demand_color = {
                "high": "#4ade80",
                "medium": "#fbbf24",
                "low": "#f87171",
            }.get(mi.get("market_demand", "medium"), "#a78bfa")
            st.markdown(
                f'<div class="market-card"><div style="color:#a78bfa;font-size:0.85rem;margin-bottom:4px">MARKET DEMAND</div><div style="font-size:1.6rem;font-weight:700;color:{demand_color}">{mi.get("market_demand", "?").upper()}</div></div>',
                unsafe_allow_html=True,
            )
        with col2:
            salary = mi.get("salary_range_inr_lpa", {})
            if salary:
                st.markdown(
                    f'<div class="market-card"><div style="color:#a78bfa;font-size:0.85rem;margin-bottom:4px">SALARY RANGE (INR LPA)</div><div style="font-size:1.4rem;font-weight:700;color:#e2e2ff">Rs {salary.get("min", "?")} - {salary.get("max", "?")} LPA</div></div>',
                    unsafe_allow_html=True,
                )

        candidate_salary = mi.get("candidate_salary_estimate_inr_lpa", {})
        if candidate_salary:
            st.markdown(
                f"**Candidate Salary Estimate:** Rs {candidate_salary.get('min', '?')} - {candidate_salary.get('max', '?')} LPA ({mi.get('candidate_level_for_role', '?').upper()} level)"
            )
        if mi.get("salary_reasoning"):
            st.markdown(f"**Salary Reasoning:** {mi.get('salary_reasoning')}")
        if mi.get("salary_caution"):
            st.markdown(f"**Salary Caution:** {mi.get('salary_caution')}")

        st.markdown(f"**🎯 Market Insight:** {mi.get('red_ocean_vs_blue_ocean', '')}")
        st.markdown(f"**💡 Stand-Out Tip:** {mi.get('market_tip', '')}")
        st.markdown(f"**⚡ Fastest Fix:** {mi.get('fastest_to_learn_blocker', '')}")

        col3, col4 = st.columns(2)
        with col3:
            st.markdown("**📈 Trending Skills in 2025**")
            for skill in mi.get("trending_skills_in_domain", []):
                st.markdown(f"- 🔥 {skill}")
        with col4:
            st.markdown("**🎯 Interview Focus Areas**")
            for skill in mi.get("interview_focus_areas", []):
                st.markdown(f"- 🎯 {skill}")

        st.markdown("**🏆 What Hired Candidates Have**")
        for item in mi.get("competitive_candidates_have", []):
            st.markdown(f"- ✅ {item}")

with tab4:
    st.markdown("### 💬 Chat with Your Resume")
    st.markdown("*Powered by OpenAI + RAG - conversational and specific*")

    for msg in st.session_state.chat_history:
        css = "chat-user" if msg["role"] == "user" else "chat-bot"
        icon = "👤" if msg["role"] == "user" else "🤖"
        st.markdown(
            f'<div class="{css}">{icon} {msg["content"]}</div>',
            unsafe_allow_html=True,
        )

    user_input = st.chat_input("Ask about your resume, skills, or career...")
    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        with st.spinner("Thinking..."):
            resp = requests.post(
                f"{API_BASE}/chat",
                json={
                    "session_id": st.session_state.session_id,
                    "message": user_input,
                    "chat_history": st.session_state.chat_history[:-1],
                },
            )
            if resp.status_code == 200:
                st.session_state.chat_history.append(
                    {"role": "assistant", "content": resp.json()["response"]}
                )
            else:
                st.error("Chat failed.")
        st.rerun()
