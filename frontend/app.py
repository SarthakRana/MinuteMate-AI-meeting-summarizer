import streamlit as st
import requests
import pandas as pd
from datetime import date

BACKEND = "http://localhost:8000"

st.set_page_config(page_title="Meeting Summarizer (OpenAI Only)", layout="wide")
st.title("MinuteMate - Your AI-Powered Meeting Summarizer")

with st.sidebar:
    st.caption("Backend URL")
    backend_url = st.text_input("Backend", BACKEND)
    st.markdown("---")
    if st.button("Load Sample"):
        try:
            sample = open("../samples/example_transcript.txt","r",encoding="utf-8").read()
            st.session_state["tx"] = sample
            st.success("Sample loaded")
        except Exception:
            st.warning("Could not load sample file.")

col1, col2 = st.columns([3,2])

with col1:
    tx = st.text_area("Paste transcript with speaker tags (e.g., `Alex:`)", 
                      value=st.session_state.get("tx",""), height=300)
    attendees = st.text_input("Attendees (optional, comma-separated)", "")
    mdate = st.date_input("Meeting date", value=date.today())
    title = st.text_input("Meeting title", "Team Sync")
    if st.button("Analyze"):
        payload = {
            "title": title,
            "transcript": tx,
            "attendees": [a.strip() for a in attendees.split(",") if a.strip()],
            "meeting_date": str(mdate)
        }
        with st.spinner("Analyzing..."):
            r = requests.post(f"{backend_url}/analyze", json=payload, timeout=180)
        if r.status_code != 200:
            st.error(f"Backend error: {r.text}")
        else:
            st.session_state["result"] = r.json()

if "result" in st.session_state:
    res = st.session_state["result"]
    with col1:
        st.subheader("Summary")
        for b in res["summary"]["bullets"]:
            st.markdown(f"- {b}")
        st.markdown("**Action Items**")
        if res["summary"]["action_items"]:
            for a in res["summary"]["action_items"]:
                st.markdown(f"- {a}")
        else:
            st.caption("No action items detected.")
        st.markdown("**Risks**")
        if res["summary"]["risks"]:
            for r in res["summary"]["risks"]:
                st.markdown(f"- {r}")
        else:
            st.caption("None identified.")

    with col2:
        st.subheader("Sentiment by Speaker (-1..+1)")
        if res["sentiment"]["by_speaker"]:
            df = pd.DataFrame({
                "speaker": list(res["sentiment"]["by_speaker"].keys()),
                "sentiment": list(res["sentiment"]["by_speaker"].values())
            })
            st.bar_chart(df.set_index("speaker"))
            st.caption(f"Overall: {round(res['sentiment']['overall'],2)}")
        else:
            st.caption("No speaker tags detected; sentiment unavailable.")

        st.markdown("---")
        # Markdown export
        md = ["# Meeting Summary", f"**Title:** {res['title']}"]
        md.append("\n## Summary")
        md.extend([f"- {b}" for b in res["summary"]["bullets"]])
        md.append("\n## Action Items")
        if res["summary"]["action_items"]:
            md.extend([f"- {a}" for a in res["summary"]["action_items"]])
        else:
            md.append("- (none)")
        md.append("\n## Sentiment")
        if res["sentiment"]["by_speaker"]:
            for k,v in res["sentiment"]["by_speaker"].items():
                md.append(f"- {k}: {round(v,2)}")
        else:
            md.append("- (not available)")
        md_text = "\n".join(md)
        st.download_button("Download Markdown", md_text, file_name="meeting_summary.md", mime="text/markdown")
