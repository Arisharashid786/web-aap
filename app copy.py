import streamlit as st
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from datetime import date

st.set_page_config(page_title="ThoraxCare AI",page_icon="🫁",layout="wide")
ROOT=Path(__file__).parent
DATA_FILE=ROOT/"patients.csv"
MODEL_FILE=ROOT/"thoracic_model.pkl"

st.markdown("""
<style>
.stApp{background:#f4f7f9;color:#18313d}
[data-testid="stSidebar"]{background:#123b4b}
[data-testid="stSidebar"] *{color:#fff}
div[data-testid="stMetric"]{background:white;padding:16px;border:1px solid #e2e9ed;border-radius:12px}
</style>
""",unsafe_allow_html=True)

# Demo directory records are illustrative, not records from the UCI dataset.
INITIAL=[
 {"Patient":"Demo Patient 01","Patient ID":"TH-1048","Age":64,"Clinical Focus":"Pulmonary nodule","Status":"Scheduled","Procedure":"VATS lobectomy","Date":str(date.today())},
 {"Patient":"Demo Patient 02","Patient ID":"TH-1045","Age":58,"Clinical Focus":"Post-VATS lobectomy","Status":"Recovery","Procedure":"Lobectomy","Date":str(date.today())},
 {"Patient":"Demo Patient 03","Patient ID":"TH-1042","Age":47,"Clinical Focus":"Pleural effusion","Status":"Review","Procedure":"Pleural procedure","Date":str(date.today())},
 {"Patient":"Demo Patient 04","Patient ID":"TH-1039","Age":71,"Clinical Focus":"Lung mass assessment","Status":"Critical","Procedure":"Assessment","Date":str(date.today())},
]
def load_patients():
    if DATA_FILE.exists(): return pd.read_csv(DATA_FILE).fillna("").to_dict("records")
    return INITIAL.copy()
def save_patients(rows): pd.DataFrame(rows).to_csv(DATA_FILE,index=False)
if "patients" not in st.session_state: st.session_state.patients=load_patients()
patients=st.session_state.patients

with st.sidebar:
    st.markdown("## 🫁 ThoraxCare")
    st.caption("Clinical Workspace • AI Demo")
    page=st.radio("Workspace",["Dashboard","Patients","Surgery Board","Follow-up","Clinical Notes","Reports","AI Prediction","Settings"])
    st.divider()
    st.caption("Prototype only ")

st.title("ThoraxCare | "+page)
st.caption("Thoracic surgery workspace — educational prototype")

if page=="Dashboard":
    c1,c2,c3,c4=st.columns(4)
    c1.metric("Directory Patients",len(patients))
    c2.metric("Scheduled",sum(p.get("Status")=="Scheduled" for p in patients))
    c3.metric("Recovery",sum(p.get("Status")=="Recovery" for p in patients))
    c4.metric("Needs Review",sum(p.get("Status") in ["Review","Critical"] for p in patients))
    st.subheader("Recent patient directory")
    st.dataframe(pd.DataFrame(patients),use_container_width=True,hide_index=True)
    st.info("The displayed patient names and IDs are fictional demo entries.")
elif page=="Patients":
    st.subheader("Register patient (demo directory)")
    with st.form("new_patient"):
        a,b,c=st.columns(3)
        name=a.text_input("Patient name")
        pid=b.text_input("Patient ID")
        age=c.number_input("Age",1,120,50)
        focus=a.text_input("Clinical focus")
        status=b.selectbox("Status",["Scheduled","Recovery","Review","Critical"])
        procedure=c.text_input("Procedure")
        submitted=st.form_submit_button("Save patient")
    if submitted:
        if not name.strip() or not pid.strip() or not focus.strip(): st.error("Name, ID and clinical focus are required.")
        elif any(str(p["Patient ID"]).lower()==pid.lower() for p in patients): st.error("Patient ID already exists.")
        else:
            patients.append({"Patient":name,"Patient ID":pid,"Age":age,"Clinical Focus":focus,"Status":status,"Procedure":procedure,"Date":str(date.today())})
            save_patients(patients); st.success("Demo patient saved."); st.rerun()
    st.subheader("Patient Directory")
    q=st.text_input("Search name, ID or clinical focus")
    df=pd.DataFrame(patients)
    if not df.empty and q:
        df=df[df.astype(str).apply(lambda col:col.str.contains(q,case=False,na=False)).any(axis=1)]
    st.dataframe(df,use_container_width=True,hide_index=True)
    if not df.empty:
        selected=st.selectbox("Select patient to remove (demo only)",["—"]+df["Patient ID"].astype(str).tolist())
        if selected!="—" and st.button("Delete selected demo patient"):
            st.session_state.patients=[p for p in patients if str(p["Patient ID"])!=selected]
            save_patients(st.session_state.patients); st.rerun()
elif page=="Surgery Board":
    st.subheader("Procedure schedule (sample entries)")
    surgeries=[
      ["Demo Patient 01","VATS Lobectomy","09:30","OT 02","Scheduled"],
      ["Demo Patient 03","Bronchoscopy Review","12:15","Procedure Suite","Scheduled"],
      ["Demo Patient 04","Pleural Procedure","14:00","OT 01","Review"]]
    st.dataframe(pd.DataFrame(surgeries,columns=["Patient","Procedure","Time","Location","Status"]),use_container_width=True,hide_index=True)
    st.caption("Replace these examples with your institution's authorized schedule.")
elif page=="Follow-up":
    st.subheader("Follow-up tracker")
    with st.form("followup"):
        patient=st.selectbox("Patient", [p["Patient ID"]+" — "+p["Patient"] for p in patients] or ["No patients"])
        focus=st.text_input("Follow-up focus","Post-operative review")
        due=st.date_input("Due date")
        status=st.selectbox("Status",["Open","Pending","Scheduled","Completed"])
        save=st.form_submit_button("Add follow-up")
    if save:
        st.session_state.setdefault("followups",[]).append({"Patient":patient,"Focus":focus,"Due":str(due),"Status":status})
        st.success("Follow-up added for this session.")
    st.dataframe(pd.DataFrame(st.session_state.get("followups",[
        {"Patient":"Demo Patient 02","Focus":"Post-operative review","Due":"2026-11-14","Status":"Open"},
        {"Patient":"Demo Patient 03","Focus":"Imaging review","Due":"2026-11-16","Status":"Pending"}])),use_container_width=True,hide_index=True)
elif page=="Clinical Notes":
    st.subheader("Clinical note drafts")
    with st.form("note"):
        patient=st.text_input("Patient name / ID")
        typ=st.selectbox("Note type",["Clinic Review","Post-operative Review","Pre-operative Assessment","MDT Summary"])
        note=st.text_area("Draft note",height=220)
        save=st.form_submit_button("Save draft")
    if save:
        st.session_state.setdefault("notes",[]).append({"Patient":patient,"Type":typ,"Note":note,"Saved":str(date.today())})
        st.success("Draft stored in this browser session only.")
    if st.session_state.get("notes"): st.dataframe(pd.DataFrame(st.session_state.notes),use_container_width=True)
    st.warning("Do not enter identifiable patient data. Notes are not persisted securely.")
elif page=="Reports":
    st.subheader("Service overview")
    x,y,z=st.columns(3); x.metric("Patients in demo directory",len(patients)); y.metric("Surgery examples",3); z.metric("Follow-up items",len(st.session_state.get("followups",[])))
    if patients: st.bar_chart(pd.Series([p["Status"] for p in patients]).value_counts())
elif page=="AI Prediction":
    st.subheader("One-year postoperative outcome model")
    st.warning("Research/education only. Not validated for clinical use; do not use for diagnosis, prognosis, triage, or treatment.")
    st.info("Bundled PKL is a synthetic placeholder so the interface runs immediately. Run train_model.py with internet access to replace it with a model trained on the authentic UCI data.")
    if MODEL_FILE.exists():
        with open(MODEL_FILE,"rb") as f: model=pickle.load(f)
        with st.form("prediction"):
            st.caption("Enter preoperative variables from the UCI dataset schema")
            age=st.slider("AGE (years)",21,87,55)
            fvc=st.number_input("PRE4 — FVC",1.4,6.3,3.3,0.1)
            fev1=st.number_input("PRE5 — FEV1",0.96,86.3,4.6,0.1)
            dgn=st.selectbox("DGN",["DGN1","DGN2","DGN3","DGN4","DGN5","DGN6","DGN8"])
            perf=st.selectbox("PRE6 (performance)",["PRZ0","PRZ1","PRZ2"])
            size=st.selectbox("PRE14 (tumor size)",["OC11","OC12","OC13","OC14"])
            bools={}
            labels={"PRE7":"Pain","PRE8":"Hemoptysis","PRE9":"Dyspnea","PRE10":"Cough","PRE11":"Weakness","PRE17":"Type 2 diabetes","PRE19":"MI within 6 months","PRE25":"Peripheral arterial disease","PRE30":"Smoking","PRE32":"Asthma"}
            for key,label in labels.items(): bools[key]=st.selectbox(label,["F","T"],index=0)
            run=st.form_submit_button("Estimate model output")
        if run:
            row=pd.DataFrame([{"DGN":dgn,"PRE4":fvc,"PRE5":fev1,"PRE6":perf,"PRE14":size,"AGE":age,**bools}])
            pred=model.predict(row)[0]
            proba=model.predict_proba(row)[0]
            # Target is Risk1Y=T (Survival within one year), F (survival beyond one year)
            st.metric("Predicted class", "Risk1Y = "+str(pred))
            st.write("Model probability by class:",dict(zip(model.classes_.astype(str),np.round(proba,3))))
            st.caption("This output is a model demonstration, not a patient-specific clinical risk estimate.")
    else: st.error("Model file missing. Run `python train_model.py` to download the real UCI dataset and train the model.")
elif page=="Settings":
    st.subheader("Workspace settings")
    doctor=st.text_input("Clinician display name","Dr. Alex Morgan")
    role=st.selectbox("Role",["Thoracic Surgeon","Consultant","Clinical Fellow"])
    if st.button("Save settings"): st.session_state["doctor"]=doctor; st.session_state["role"]=role; st.success("Settings saved for this session.")
    st.caption("Settings are session-only in this prototype.")

st.divider()
st.caption("ThoraxCare AI prototype • Data governance, privacy, validation and clinical review required before deployment.")
