"""Research scenarios and synthetic datasets for QuantAI v4 training data.

Each scenario is a fictional dataset with realistic variable names from public
health and clinical research. Datasets are generated, not collected: they are
only used to render the exact context QuantAI's browser sends to the model and
to check that every expected action runs in QuantAI's real engine.

Training and evaluation scenarios are disjoint (see SPLIT), so evaluation
measures generalisation to new research settings, not recall of training ones.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Var:
    key: str                 # internal key used by templates
    header: str              # column header as it appears in the CSV
    kind: str                # age cont count bin cat ord expo time event clus
    mentions: list[str]      # how a user might refer to it in a message
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class Scenario:
    sid: str
    split: str               # "train" or "eval"
    title: str               # used as the dataset source name
    variables: list[Var]
    n: tuple[int, int] = (180, 420)
    survival: list[tuple[str, str]] = field(default_factory=list)   # (time key, event key)
    rates: list[tuple[str, str]] = field(default_factory=list)      # (count key, exposure key)
    paired: list[tuple[str, str]] = field(default_factory=list)     # (before key, after key)
    series: bool = False     # one row per period, for forecasting
    builtin: bool = False    # QuantAI's own example dataset (already loaded in the app)

    def var(self, key: str) -> Var:
        for v in self.variables:
            if v.key == key:
                return v
        raise KeyError(key)

    def by_kind(self, *kinds: str) -> list[Var]:
        return [v for v in self.variables if v.kind in kinds]


def V(key, header, kind, mentions, **params) -> Var:
    return Var(key, header, kind, mentions if isinstance(mentions, list) else [mentions], params)


YN = ["No", "Yes"]
SEX = ["Female", "Male"]
LMH = ["Low", "Medium", "High"]
MSEV = ["Mild", "Moderate", "Severe"]
EDU = ["None", "Primary", "Secondary", "Tertiary"]
FREQ = ["Never", "Rarely", "Sometimes", "Often", "Always"]
HEALTH = ["Poor", "Fair", "Good", "Very good"]
AGREE = ["Strongly disagree", "Disagree", "Neutral", "Agree", "Strongly agree"]
SATIS = ["Very dissatisfied", "Dissatisfied", "Neutral", "Satisfied", "Very satisfied"]


def clusters(prefix: str, n: int) -> list[str]:
    return [f"{prefix} {i:02d}" for i in range(1, n + 1)]


# ---------------------------------------------------------------------------
# Training scenarios
# ---------------------------------------------------------------------------
TRAIN = [
    Scenario("malaria_hh", "train", "malaria_household_survey.csv", [
        V("age", "Age (years)", "age", ["age"], lo=1, hi=80),
        V("sex", "Sex", "bin", ["sex", "gender"], levels=SEX),
        V("net", "Bed net use", "bin", ["bed net use", "net use", "sleeping under a bed net"], levels=YN, p=0.6),
        V("rdt", "Malaria RDT result", "bin", ["malaria RDT result", "malaria positivity", "a positive malaria test"], levels=["Negative", "Positive"], p=0.3),
        V("hhsize", "Household size", "count", ["household size"], lam=6),
        V("dist", "Distance to facility (km)", "cont", ["distance to the facility", "distance to the health facility"], mean=6, sd=3, lo=0.2),
        V("district", "District", "cat", ["district"], levels=["Ho", "Keta", "Hohoe", "Kpando", "Akatsi"]),
        V("wealth", "Wealth index", "ord", ["wealth index", "wealth"], levels=LMH),
        V("episodes", "Malaria episodes", "count", ["malaria episodes", "number of malaria episodes"], lam=4),
        V("pm", "Person-months of follow-up", "expo", ["person-months of follow-up", "follow-up time"], lo=3, hi=24),
        V("community", "Community", "clus", ["community"], k=24),
    ], rates=[("episodes", "pm")]),

    Scenario("maternal_anaemia", "train", "anc_anaemia_study.csv", [
        V("age", "Maternal age", "age", ["maternal age", "age"], lo=15, hi=45),
        V("trimester", "Trimester", "ord", ["trimester"], levels=["First", "Second", "Third"]),
        V("parity", "Parity", "count", ["parity"], lam=3),
        V("hb", "Haemoglobin (g/dL)", "cont", ["haemoglobin", "Hb", "haemoglobin level"], mean=11, sd=1.5, lo=5),
        V("anaemia", "Anaemia", "bin", ["anaemia", "being anaemic"], levels=YN, p=0.35),
        V("anc", "ANC visits", "count", ["ANC visits", "number of antenatal visits"], lam=4),
        V("iron", "Iron supplementation", "bin", ["iron supplementation", "taking iron"], levels=YN, p=0.55),
        V("residence", "Residence", "bin", ["residence", "rural or urban residence"], levels=["Rural", "Urban"]),
        V("facility", "Health facility", "clus", ["health facility", "facility"], k=18),
    ]),

    Scenario("hiv_art", "train", "art_cohort.csv", [
        V("age", "Age at ART start", "age", ["age at ART start", "age"], lo=18, hi=70),
        V("sex", "Sex", "bin", ["sex"], levels=SEX),
        V("vls", "Viral load suppressed", "bin", ["viral suppression", "viral load suppression"], levels=YN, p=0.75),
        V("cd4", "Baseline CD4 count", "cont", ["baseline CD4", "CD4 count"], mean=350, sd=150, lo=10, dec=0),
        V("months", "Months on ART", "time", ["time on ART", "months on ART", "follow-up time"], scale=24),
        V("died", "Died", "event", ["death", "mortality", "dying"], p=0.25),
        V("regimen", "Regimen", "cat", ["regimen", "ART regimen"], levels=["TDF/3TC/DTG", "TDF/3TC/EFV", "AZT/3TC/NVP"]),
        V("adherence", "Adherence", "ord", ["adherence"], levels=["Poor", "Fair", "Good"]),
        V("missed", "Missed appointments", "count", ["missed appointments"], lam=3),
        V("clinic", "ART clinic", "clus", ["ART clinic", "clinic"], k=16),
    ], survival=[("months", "died")]),

    Scenario("tb_outcomes", "train", "tb_register.csv", [
        V("age", "Age", "age", ["age"], lo=15, hi=85),
        V("sex", "Sex", "bin", ["sex", "gender"], levels=SEX),
        V("hiv", "HIV status", "bin", ["HIV status", "HIV"], levels=["Negative", "Positive"], p=0.25),
        V("outcome", "Treatment outcome", "cat", ["treatment outcome"], levels=["Cured", "Completed", "Failed", "Died"], probs=[0.45, 0.3, 0.12, 0.13]),
        V("days", "Days to culture conversion", "time", ["time to culture conversion", "days to conversion"], scale=60),
        V("converted", "Culture converted", "event", ["culture conversion", "conversion"], p=0.7),
        V("weightgain", "Weight gain (kg)", "cont", ["weight gain"], mean=3, sd=2, lo=-4),
        V("smear", "Smear grade", "ord", ["smear grade"], levels=LMH),
        V("site", "Treatment site", "clus", ["treatment site", "site"], k=14),
    ], survival=[("days", "converted")]),

    Scenario("child_nutrition", "train", "child_growth_survey.csv", [
        V("agem", "Child age (months)", "cont", ["child's age", "age in months", "age"], mean=30, sd=14, lo=6, hi=59, dec=0),
        V("sex", "Sex of child", "bin", ["sex", "child's sex"], levels=SEX),
        V("stunted", "Stunted", "bin", ["stunting", "being stunted"], levels=YN, p=0.3),
        V("haz", "Height-for-age z-score", "cont", ["height-for-age z-score", "HAZ"], mean=-1.2, sd=1.1, lo=-5, hi=3, dec=2),
        V("medu", "Mother's education", "ord", ["mother's education", "maternal education"], levels=EDU),
        V("diarrhoea", "Diarrhoea episodes", "count", ["diarrhoea episodes", "number of diarrhoea episodes"], lam=3),
        V("water", "Water source", "cat", ["water source", "drinking water source"], levels=["Borehole", "Pipe", "River", "Well"]),
        V("village", "Village", "clus", ["village"], k=22),
    ]),

    Scenario("htn_clinic", "train", "hypertension_followup.csv", [
        V("age", "Age", "age", ["age"], lo=30, hi=85),
        V("sex", "Sex", "bin", ["sex"], levels=SEX),
        V("sbp0", "SBP baseline", "cont", ["baseline systolic BP", "systolic BP at baseline"], mean=150, sd=15, lo=95, dec=0),
        V("sbp6", "SBP 6 months", "cont", ["systolic BP at 6 months", "6-month systolic BP"], mean=138, sd=16, lo=90, dec=0),
        V("controlled", "BP controlled", "bin", ["BP control", "blood pressure control"], levels=YN, p=0.45),
        V("meds", "Medication adherence", "ord", ["medication adherence", "adherence"], levels=FREQ),
        V("bmi", "BMI", "cont", ["BMI", "body mass index"], mean=28, sd=5, lo=16, hi=50),
        V("clinic", "Clinic", "clus", ["clinic"], k=20),
        V("visits", "Clinic visits", "count", ["clinic visits", "number of clinic visits"], lam=4),
    ], paired=[("sbp0", "sbp6")]),

    Scenario("diabetes", "train", "diabetes_registry.csv", [
        V("age", "Age (years)", "age", ["age"], lo=25, hi=85),
        V("sex", "Gender", "bin", ["gender", "sex"], levels=SEX),
        V("hba1c", "HbA1c (%)", "cont", ["HbA1c"], mean=8, sd=1.6, lo=4.5),
        V("fbg", "Fasting glucose (mmol/L)", "cont", ["fasting glucose", "fasting blood glucose"], mean=8.5, sd=2.5, lo=3),
        V("controlled", "Glycaemic control", "bin", ["glycaemic control", "good glucose control"], levels=YN, p=0.4),
        V("duration", "Diabetes duration (years)", "cont", ["diabetes duration", "years with diabetes"], mean=8, sd=5, lo=0.5),
        V("ulcer", "Foot ulcer", "bin", ["foot ulcer", "having a foot ulcer"], levels=YN, p=0.15),
        V("admissions", "Number of admissions", "count", ["hospital admissions", "number of admissions"], lam=6),
        V("py", "Patient-years at risk", "expo", ["patient-years at risk", "years at risk"], lo=0.5, hi=6),
        V("treatment", "Treatment type", "cat", ["treatment type", "treatment"], levels=["Diet only", "Oral agents", "Insulin"]),
    ], rates=[("admissions", "py")]),

    Scenario("immunisation", "train", "immunisation_coverage.csv", [
        V("agem", "Child age (months)", "cont", ["child's age", "age"], mean=18, sd=4, lo=12, hi=23, dec=0),
        V("sex", "Sex", "bin", ["sex"], levels=SEX),
        V("full", "Fully immunised", "bin", ["full immunisation", "being fully immunised"], levels=YN, p=0.7),
        V("dist", "Distance to clinic (km)", "cont", ["distance to the clinic", "distance"], mean=5, sd=3, lo=0.1),
        V("medu", "Mother's education", "ord", ["mother's education"], levels=EDU),
        V("siblings", "Number of siblings", "count", ["number of siblings", "siblings"], lam=3),
        V("zone", "Health zone", "clus", ["health zone", "zone"], k=15),
        V("outreach", "Outreach visits", "count", ["outreach visits"], lam=3),
    ]),

    Scenario("wash_school", "train", "school_wash_survey.csv", [
        V("age", "Pupil age", "age", ["pupil age", "age"], lo=8, hi=15),
        V("sex", "Sex", "bin", ["sex", "gender"], levels=SEX),
        V("knowledge", "Handwashing knowledge score", "cont", ["handwashing knowledge score", "knowledge score"], mean=60, sd=15, lo=0, hi=100, dec=0),
        V("diarrhoea", "Diarrhoea past 2 weeks", "bin", ["diarrhoea in the past two weeks", "recent diarrhoea"], levels=YN, p=0.2),
        V("absent", "Days absent", "count", ["days absent", "absenteeism"], lam=4),
        V("school", "School", "clus", ["school"], k=20),
        V("latrine", "Latrine access", "bin", ["latrine access"], levels=YN, p=0.6),
        V("cls", "Class", "cat", ["class"], levels=["P4", "P5", "P6"]),
    ]),

    Scenario("neonatal", "train", "neonatal_unit.csv", [
        V("bw", "Birth weight (g)", "cont", ["birth weight"], mean=2900, sd=550, lo=700, dec=0),
        V("ga", "Gestational age (weeks)", "cont", ["gestational age"], mean=38, sd=2.2, lo=24, hi=43),
        V("sex", "Sex of baby", "bin", ["baby's sex", "sex"], levels=SEX),
        V("lbw", "Low birth weight", "bin", ["low birth weight", "LBW"], levels=YN, p=0.18),
        V("mode", "Mode of delivery", "cat", ["mode of delivery", "delivery mode"], levels=["Vaginal", "Assisted", "Caesarean"]),
        V("mage", "Maternal age", "age", ["maternal age", "mother's age"], lo=15, hi=45),
        V("anc", "ANC visits", "count", ["ANC visits", "antenatal visits"], lam=4),
        V("stay", "Days in unit", "time", ["length of stay", "days in the unit"], scale=8),
        V("discharged", "Discharged alive", "event", ["discharge", "being discharged alive"], p=0.8),
        V("hospital", "Hospital", "clus", ["hospital"], k=12),
    ], survival=[("stay", "discharged")]),

    Scenario("mental_health", "train", "community_mental_health.csv", [
        V("age", "Age", "age", ["age"], lo=18, hi=75),
        V("sex", "Sex", "bin", ["sex"], levels=SEX),
        V("phq9", "PHQ-9 score", "cont", ["PHQ-9 score", "depression score"], mean=9, sd=5, lo=0, hi=27, dec=0),
        V("depressed", "Depression", "bin", ["depression", "being depressed"], levels=YN, p=0.25),
        V("anxiety", "Anxiety severity", "ord", ["anxiety severity", "anxiety"], levels=MSEV),
        V("employment", "Employment status", "cat", ["employment status", "employment"], levels=["Employed", "Self-employed", "Unemployed", "Student"]),
        V("sleep", "Sleep hours", "cont", ["sleep hours", "hours of sleep"], mean=6.5, sd=1.3, lo=2, hi=12),
        V("sessions", "Counselling sessions", "count", ["counselling sessions"], lam=4),
        V("community", "Community", "clus", ["community"], k=18),
    ]),

    Scenario("hai", "train", "hospital_infections.csv", [
        V("ward", "Ward", "cat", ["ward"], levels=["Medical", "Surgical", "Paediatric", "Maternity"]),
        V("age", "Patient age", "age", ["patient age", "age"], lo=1, hi=90),
        V("los", "Length of stay (days)", "cont", ["length of stay"], mean=7, sd=4, lo=1),
        V("infections", "Infections", "count", ["infections", "number of infections", "infection counts"], lam=4),
        V("ptdays", "Patient days", "expo", ["patient-days", "patient days"], lo=20, hi=400),
        V("ratio", "Nurse-to-patient ratio", "cont", ["nurse-to-patient ratio", "staffing ratio"], mean=0.25, sd=0.08, lo=0.05, dec=2),
        V("hygiene", "Hand hygiene compliance (%)", "cont", ["hand hygiene compliance"], mean=65, sd=15, lo=10, hi=100, dec=0),
        V("hospital", "Hospital", "clus", ["hospital"], k=12),
    ], rates=[("infections", "ptdays")]),

    Scenario("road_traffic", "train", "trauma_registry.csv", [
        V("age", "Age", "age", ["age"], lo=5, hi=80),
        V("sex", "Sex", "bin", ["sex"], levels=SEX),
        V("severity", "Injury severity", "ord", ["injury severity", "severity"], levels=MSEV),
        V("helmet", "Helmet use", "bin", ["helmet use", "wearing a helmet"], levels=YN, p=0.4),
        V("speed", "Estimated speed (km/h)", "cont", ["speed", "estimated speed"], mean=55, sd=20, lo=5, dec=0),
        V("vehicle", "Vehicle type", "cat", ["vehicle type"], levels=["Motorbike", "Car", "Bus", "Pedestrian"]),
        V("hdays", "Days in hospital", "time", ["days in hospital", "time in hospital"], scale=10),
        V("died", "Died", "event", ["death", "in-hospital death"], p=0.12),
    ], survival=[("hdays", "died")]),

    Scenario("vaccine_hesitancy", "train", "vaccine_attitudes.csv", [
        V("age", "Age", "age", ["age"], lo=18, hi=80),
        V("sex", "Sex", "bin", ["sex", "gender"], levels=SEX),
        V("vaccinated", "Vaccinated", "bin", ["vaccination", "being vaccinated"], levels=YN, p=0.55),
        V("hesitancy", "Vaccines are safe", "ord", ["agreement that vaccines are safe", "the vaccine safety item"], levels=AGREE),
        V("edu", "Education level", "ord", ["education", "education level"], levels=EDU),
        V("occupation", "Occupation", "cat", ["occupation"], levels=["Farmer", "Trader", "Civil servant", "Student", "Other"]),
        V("trust", "Trust in health system score", "cont", ["trust score", "trust in the health system"], mean=3.2, sd=0.9, lo=1, hi=5),
        V("region", "Region", "cat", ["region"], levels=["Volta", "Ashanti", "Northern", "Greater Accra"]),
    ]),

    Scenario("cancer", "train", "cervical_cancer_cohort.csv", [
        V("age", "Age at diagnosis", "age", ["age at diagnosis", "age"], lo=25, hi=80),
        V("stage", "FIGO stage", "cat", ["stage", "FIGO stage"], levels=["I", "II", "III", "IV"]),
        V("arm", "Treatment arm", "cat", ["treatment arm", "treatment group"], levels=["Chemoradiation", "Radiation only", "Surgery"]),
        V("months", "Survival time (months)", "time", ["survival time", "time"], scale=30),
        V("died", "Death", "event", ["death", "dying"], p=0.45),
        V("tumour", "Tumour size (cm)", "cont", ["tumour size"], mean=4, sd=1.5, lo=0.5),
        V("hiv", "HIV positive", "bin", ["HIV status", "HIV"], levels=YN, p=0.3),
        V("hospital", "Treating hospital", "clus", ["hospital", "treating hospital"], k=10),
    ], survival=[("months", "died")]),

    Scenario("occupational", "train", "factory_hearing_study.csv", [
        V("age", "Age", "age", ["age"], lo=18, hi=65),
        V("sex", "Sex", "bin", ["sex"], levels=SEX),
        V("years", "Years employed", "cont", ["years employed", "length of employment"], mean=9, sd=6, lo=0.5),
        V("hearing", "Hearing loss", "bin", ["hearing loss"], levels=YN, p=0.3),
        V("noise", "Noise level (dB)", "cont", ["noise level", "noise exposure"], mean=85, sd=8, lo=55, dec=0),
        V("job", "Job category", "cat", ["job category", "job type"], levels=["Machine operator", "Maintenance", "Office", "Packing"]),
        V("sickdays", "Sick days", "count", ["sick days", "number of sick days"], lam=5),
        V("pyears", "Person-years employed", "expo", ["person-years of employment", "person-years"], lo=0.5, hi=20),
        V("factory", "Factory", "clus", ["factory"], k=10),
    ], rates=[("sickdays", "pyears")]),

    Scenario("family_planning", "train", "fp_survey.csv", [
        V("age", "Age", "age", ["age"], lo=15, hi=49),
        V("marital", "Marital status", "cat", ["marital status"], levels=["Married", "Cohabiting", "Single", "Divorced"]),
        V("modern", "Modern contraceptive use", "bin", ["modern contraceptive use", "using modern contraception"], levels=YN, p=0.35),
        V("children", "Number of children", "count", ["number of children", "parity"], lam=3),
        V("edu", "Education", "ord", ["education"], levels=EDU),
        V("residence", "Residence", "bin", ["residence", "urban or rural residence"], levels=["Rural", "Urban"]),
        V("support", "Partner support", "ord", ["partner support"], levels=LMH),
        V("district", "District", "cat", ["district"], levels=["Ho", "Hohoe", "Keta", "Kpando"]),
    ]),

    Scenario("sickle_cell", "train", "sickle_cell_clinic.csv", [
        V("age", "Age", "age", ["age"], lo=2, hi=40),
        V("sex", "Sex", "bin", ["sex"], levels=SEX),
        V("crises", "Number of pain crises", "count", ["pain crises", "number of crises", "crisis counts"], lam=5),
        V("hu", "Hydroxyurea", "bin", ["hydroxyurea use", "hydroxyurea"], levels=YN, p=0.4),
        V("hb", "Haemoglobin (g/dL)", "cont", ["haemoglobin", "Hb"], mean=8, sd=1.2, lo=4),
        V("admit", "Number of admissions", "count", ["hospital admissions", "admissions"], lam=6),
        V("fu", "Follow-up years", "expo", ["follow-up years", "years of follow-up"], lo=0.5, hi=5),
        V("clinic", "Clinic", "clus", ["clinic"], k=8),
    ], rates=[("crises", "fu"), ("admit", "fu")]),

    Scenario("eye_health", "train", "cataract_programme.csv", [
        V("age", "Age", "age", ["age"], lo=40, hi=90),
        V("sex", "Sex", "bin", ["sex"], levels=SEX),
        V("acuity", "Visual acuity (logMAR)", "cont", ["visual acuity"], mean=0.8, sd=0.4, lo=0, hi=2, dec=2),
        V("cataract", "Cataract", "bin", ["cataract"], levels=YN, p=0.4),
        V("diabetes", "Diabetes", "bin", ["diabetes"], levels=YN, p=0.2),
        V("wait", "Days to surgery", "time", ["waiting time to surgery", "days to surgery"], scale=60),
        V("operated", "Operated", "event", ["surgery", "being operated"], p=0.6),
        V("district", "District", "cat", ["district"], levels=["North", "South", "East", "West"]),
    ], survival=[("wait", "operated")]),

    Scenario("elderly", "train", "older_adults_study.csv", [
        V("age", "Age", "age", ["age"], lo=60, hi=95),
        V("sex", "Sex", "bin", ["sex"], levels=SEX),
        V("frailty", "Frailty score", "cont", ["frailty score", "frailty"], mean=4, sd=2, lo=0, hi=12, dec=1),
        V("falls", "Falls in past year", "count", ["falls in the past year", "number of falls"], lam=3),
        V("alone", "Lives alone", "bin", ["living alone"], levels=YN, p=0.3),
        V("health", "Self-rated health", "ord", ["self-rated health"], levels=HEALTH),
        V("hosp", "Hospitalised", "bin", ["hospitalisation", "being hospitalised"], levels=YN, p=0.2),
        V("community", "Community", "clus", ["community"], k=16),
    ]),

    Scenario("adolescent", "train", "adolescent_health_survey.csv", [
        V("age", "Age", "age", ["age"], lo=12, hi=19),
        V("sex", "Sex", "bin", ["sex", "gender"], levels=SEX),
        V("alcohol", "Alcohol use", "bin", ["alcohol use", "drinking alcohol"], levels=YN, p=0.25),
        V("activity", "Physical activity (min/day)", "cont", ["physical activity"], mean=45, sd=25, lo=0, dec=1),
        V("school", "School", "clus", ["school"], k=18),
        V("screen", "Screen time (hours/day)", "cont", ["screen time"], mean=4, sd=2, lo=0, hi=14),
        V("wellbeing", "Wellbeing score", "cont", ["wellbeing score", "wellbeing"], mean=50, sd=10, lo=14, hi=70, dec=0),
        V("smoking", "Smoking frequency", "ord", ["smoking frequency", "how often they smoke"], levels=FREQ),
    ]),

    Scenario("pharmacy", "train", "facility_stock_audit.csv", [
        V("ftype", "Facility type", "cat", ["facility type"], levels=["CHPS", "Health centre", "Hospital"]),
        V("stockout", "Stock-out days", "count", ["stock-out days", "days out of stock"], lam=6),
        V("orders", "Monthly orders (GHS)", "cont", ["monthly orders", "order value"], mean=4000, sd=1500, lo=200, dec=0),
        V("dist", "Distance to district store (km)", "cont", ["distance to the district store", "distance"], mean=25, sd=15, lo=1),
        V("obs", "Days observed", "expo", ["days observed", "observation days"], lo=30, hi=365),
        V("region", "Region", "cat", ["region"], levels=["Volta", "Oti", "Eastern"]),
        V("staff", "Pharmacy staff", "count", ["number of pharmacy staff", "staff numbers"], lam=4),
    ], rates=[("stockout", "obs")]),

    Scenario("lab_quality", "train", "lab_turnaround.csv", [
        V("tat", "Turnaround time (hours)", "cont", ["turnaround time", "TAT"], mean=20, sd=9, lo=1),
        V("rejected", "Sample rejected", "bin", ["sample rejection", "rejected samples"], levels=YN, p=0.12),
        V("lab", "Laboratory", "clus", ["laboratory", "lab"], k=11),
        V("shift", "Shift", "cat", ["shift"], levels=["Morning", "Afternoon", "Night"]),
        V("workload", "Daily workload", "count", ["daily workload", "workload"], lam=6),
        V("experience", "Technician experience (years)", "cont", ["technician experience", "years of experience"], mean=7, sd=5, lo=0.5),
    ]),

    Scenario("adult_nutrition", "train", "adult_nutrition_survey.csv", [
        V("age", "Age", "age", ["age"], lo=18, hi=75),
        V("sex", "Sex", "bin", ["sex"], levels=SEX),
        V("bmi", "BMI (kg/m2)", "cont", ["BMI"], mean=26, sd=5, lo=15, hi=50),
        V("obese", "Obese", "bin", ["obesity", "being obese"], levels=YN, p=0.25),
        V("fruit", "Fruit and vegetable servings", "count", ["fruit and vegetable servings", "servings of fruit and vegetables"], lam=4),
        V("activity", "Physical activity level", "ord", ["physical activity level", "activity level"], levels=LMH),
        V("income", "Income group", "ord", ["income group", "income"], levels=["Lowest", "Middle", "Highest"]),
        V("waist", "Waist circumference (cm)", "cont", ["waist circumference"], mean=88, sd=12, lo=55, dec=0),
    ]),

    # forecasting series (one row per period)
    Scenario("malaria_monthly", "train", "district_malaria_monthly.csv", [
        V("month", "Month", "date", ["month"], unit="month", start=(2018, 1)),
        V("cases", "Malaria cases", "series", ["malaria cases", "monthly malaria cases"], base=900, amp=300),
        V("tests", "RDTs done", "series", ["RDTs done", "tests done"], base=2500, amp=500),
    ], n=(48, 72), series=True),
    Scenario("opd_weekly", "train", "opd_attendance_weekly.csv", [
        V("week", "Week starting", "date", ["week"], unit="week", start=(2022, 1)),
        V("opd", "OPD attendance", "series", ["OPD attendance", "outpatient attendance"], base=1200, amp=150),
    ], n=(60, 104), series=True),
    Scenario("deliveries_q", "train", "facility_deliveries_quarterly.csv", [
        V("quarter", "Quarter", "date", ["quarter"], unit="quarter", start=(2014, 1)),
        V("deliveries", "Skilled deliveries", "series", ["skilled deliveries", "deliveries"], base=400, amp=40),
    ], n=(28, 40), series=True),
]

# ---------------------------------------------------------------------------
# Held-out evaluation scenarios (never used for training)
# ---------------------------------------------------------------------------
EVAL = [
    Scenario("falls_ward", "eval", "inpatient_falls.csv", [
        V("falls", "Falls", "count", ["falls", "inpatient falls", "fall counts"], lam=4),
        V("beddays", "Bed days", "expo", ["bed-days", "bed days"], lo=50, hi=900),
        V("staffing", "Staffing ratio", "cont", ["staffing ratio", "staffing"], mean=0.3, sd=0.1, lo=0.05, dec=2),
        V("ward", "Ward", "cat", ["ward"], levels=["Medical", "Surgical", "Orthopaedic", "Geriatric"]),
        V("age", "Mean patient age", "cont", ["mean patient age", "patient age"], mean=62, sd=10, lo=20),
        V("hospital", "Hospital", "clus", ["hospital"], k=14),
    ], rates=[("falls", "beddays")]),

    Scenario("schisto", "eval", "schisto_school_survey.csv", [
        V("age", "Age (years)", "age", ["age"], lo=6, hi=16),
        V("sex", "Sex", "bin", ["sex", "gender"], levels=SEX),
        V("infected", "S. haematobium infection", "bin", ["infection", "schistosomiasis infection"], levels=["Negative", "Positive"], p=0.3),
        V("eggs", "Egg count per 10 ml", "count", ["egg count", "egg counts"], lam=6),
        V("swims", "Swims in river", "bin", ["swimming in the river", "river swimming"], levels=YN, p=0.5),
        V("school", "School", "clus", ["school"], k=16),
        V("pzq", "Received praziquantel", "bin", ["praziquantel treatment", "praziquantel"], levels=YN, p=0.6),
        V("contact", "Water contact score", "cont", ["water contact score", "water contact"], mean=10, sd=4, lo=0, dec=0),
    ]),

    Scenario("dialysis", "eval", "dialysis_cohort.csv", [
        V("age", "Age", "age", ["age"], lo=20, hi=85),
        V("sex", "Sex", "bin", ["sex"], levels=SEX),
        V("egfr", "eGFR at start", "cont", ["eGFR", "baseline eGFR"], mean=9, sd=3, lo=2),
        V("months", "Months on dialysis", "time", ["time on dialysis", "months on dialysis"], scale=20),
        V("died", "Died", "event", ["death", "mortality"], p=0.4),
        V("diabetic", "Diabetic", "bin", ["diabetes", "being diabetic"], levels=YN, p=0.35),
        V("hosps", "Hospitalisations", "count", ["hospitalisations", "number of hospitalisations"], lam=4),
        V("ptmonths", "Patient-months", "expo", ["patient-months", "patient months of follow-up"], lo=1, hi=48),
        V("centre", "Dialysis centre", "clus", ["dialysis centre", "centre"], k=9),
    ], survival=[("months", "died")], rates=[("hosps", "ptmonths")]),

    Scenario("breastfeeding", "eval", "infant_feeding_cohort.csv", [
        V("agem", "Infant age (months)", "cont", ["infant age", "age of the infant"], mean=7, sd=3, lo=0, hi=12, dec=0),
        V("ebf", "Exclusive breastfeeding", "bin", ["exclusive breastfeeding"], levels=YN, p=0.45),
        V("mage", "Mother's age", "age", ["mother's age", "maternal age"], lo=15, hi=45),
        V("medu", "Mother's education", "ord", ["mother's education", "maternal education"], levels=EDU),
        V("place", "Place of delivery", "cat", ["place of delivery", "delivery place"], levels=["Home", "Health centre", "Hospital"]),
        V("weeks", "Breastfeeding duration (weeks)", "time", ["breastfeeding duration", "duration of breastfeeding"], scale=20),
        V("stopped", "Stopped breastfeeding", "event", ["stopping breastfeeding", "cessation"], p=0.6),
        V("village", "Village", "clus", ["village"], k=15),
    ], survival=[("weeks", "stopped")]),

    Scenario("burnout", "eval", "health_worker_wellbeing.csv", [
        V("age", "Age", "age", ["age"], lo=22, hi=62),
        V("sex", "Sex", "bin", ["sex"], levels=SEX),
        V("burnout", "Burnout score", "cont", ["burnout score", "burnout"], mean=45, sd=12, lo=0, hi=100, dec=0),
        V("burned", "High burnout", "bin", ["high burnout", "being burned out"], levels=YN, p=0.35),
        V("satisfaction", "Job satisfaction", "ord", ["job satisfaction"], levels=SATIS),
        V("shift", "Shift type", "cat", ["shift type", "shift pattern"], levels=["Day", "Night", "Rotating"]),
        V("overtime", "Overtime hours per week", "cont", ["overtime", "overtime hours"], mean=8, sd=5, lo=0),
        V("department", "Department", "clus", ["department"], k=12),
    ]),

    Scenario("otitis", "eval", "childhood_ear_infections.csv", [
        V("agem", "Age (months)", "cont", ["age", "child's age"], mean=30, sd=14, lo=3, hi=60, dec=0),
        V("sex", "Sex", "bin", ["sex"], levels=SEX),
        V("episodes", "Ear infection episodes", "count", ["ear infection episodes", "number of ear infections"], lam=4),
        V("daycare", "Attends daycare", "bin", ["daycare attendance", "attending daycare"], levels=YN, p=0.4),
        V("childyears", "Child-years observed", "expo", ["child-years of observation", "child-years"], lo=0.3, hi=4),
        V("smoke", "Household smoke exposure", "bin", ["household smoke exposure", "smoke exposure"], levels=YN, p=0.3),
        V("clinic", "Clinic", "clus", ["clinic"], k=10),
    ], rates=[("episodes", "childyears")]),

    Scenario("stroke_rehab", "eval", "stroke_rehabilitation.csv", [
        V("age", "Age", "age", ["age"], lo=35, hi=90),
        V("sex", "Sex", "bin", ["sex"], levels=SEX),
        V("barthel0", "Barthel index admission", "cont", ["Barthel index on admission", "admission Barthel score"], mean=40, sd=18, lo=5, hi=100, dec=1),
        V("barthel3", "Barthel index 3 months", "cont", ["Barthel index at 3 months", "3-month Barthel score"], mean=62, sd=20, lo=5, hi=100, dec=1),
        V("rehab", "Rehab hours", "cont", ["rehabilitation hours", "hours of rehab"], mean=30, sd=12, lo=0),
        V("days", "Days to discharge", "time", ["time to discharge", "days to discharge"], scale=20),
        V("discharged", "Discharged", "event", ["discharge"], p=0.85),
        V("destination", "Discharge destination", "cat", ["discharge destination"], levels=["Home", "Rehab unit", "Care home"]),
        V("hospital", "Hospital", "clus", ["hospital"], k=10),
    ], survival=[("days", "discharged")], paired=[("barthel0", "barthel3")]),

    Scenario("cholera", "eval", "cholera_line_list.csv", [
        V("age", "Age", "age", ["age"], lo=1, hi=80),
        V("sex", "Sex", "bin", ["sex"], levels=SEX),
        V("dehydration", "Dehydration", "ord", ["dehydration severity", "dehydration"], levels=MSEV),
        V("died", "Died", "bin", ["death", "dying"], levels=YN, p=0.06),
        V("recovery", "Days to recovery", "time", ["time to recovery", "days to recovery"], scale=5),
        V("recovered", "Recovered", "event", ["recovery"], p=0.85),
        V("water", "Water source", "cat", ["water source"], levels=["Borehole", "River", "Pipe", "Vendor"]),
        V("district", "District", "cat", ["district"], levels=["Ketu South", "Anloga", "Keta", "Akatsi"]),
    ], survival=[("recovery", "recovered")]),

    Scenario("cholera_weekly", "eval", "cholera_weekly.csv", [
        V("week", "Epi week", "date", ["epi week", "week"], unit="week", start=(2023, 1)),
        V("cases", "Cholera cases", "series", ["cholera cases", "weekly cholera cases"], base=60, amp=30),
    ], n=(52, 80), series=True),
    Scenario("ed_monthly", "eval", "emergency_visits_monthly.csv", [
        V("month", "Month", "date", ["month"], unit="month", start=(2017, 1)),
        V("visits", "ED visits", "series", ["emergency department visits", "ED visits"], base=3000, amp=400),
    ], n=(60, 84), series=True),
]

# QuantAI's built-in example dataset (always loaded before a user uploads data).
# Mentions map to the app's own variable names; the rows come from the app itself.
BUILTIN = Scenario("example_htn", "both", "Example: Ghana hypertension survey (fictional)", [
    V("age_years", "Age (years)", "cont", ["age"]),
    V("sex", "Sex", "bin", ["sex", "gender"]),
    V("district", "District", "cat", ["district"]),
    V("residence", "Residence", "bin", ["residence", "urban or rural residence"]),
    V("education_level", "Education level", "ord", ["education", "education level"]),
    V("salt_intake", "Salt intake", "ord", ["salt intake"]),
    V("bmi", "BMI", "cont", ["BMI"]),
    V("systolic_bp", "Systolic BP", "cont", ["systolic BP", "systolic blood pressure"]),
    V("systolic_bp_3_months", "Systolic BP 3 months", "cont", ["systolic BP at 3 months"]),
    V("hypertension", "Hypertension", "bin", ["hypertension"]),
    V("smoker", "Smoker", "bin", ["smoking", "being a smoker"]),
    V("clinic_visits_12m", "Clinic visits (12m)", "count", ["clinic visits", "number of clinic visits"]),
    V("aware_of_status_baseline", "Aware of status (baseline)", "bin", ["awareness of status at baseline"]),
    V("aware_of_status_3_months", "Aware of status (3 months)", "bin", ["awareness of status at 3 months"]),
    V("clinic", "Clinic", "clus", ["clinic"]),
    V("systolic_bp_6_months", "Systolic BP 6 months", "cont", ["systolic BP at 6 months"]),
    V("follow_up_months", "Follow-up (months)", "time", ["follow-up time", "follow-up"]),
    V("lost_to_follow_up", "Lost to follow-up", "event", ["loss to follow-up", "being lost to follow-up"]),
    V("self_rated_health", "Self-rated health", "ord", ["self-rated health"]),
    V("usual_source_of_care", "Usual source of care", "cat", ["usual source of care"]),
], survival=[("follow_up_months", "lost_to_follow_up")], paired=[("systolic_bp", "systolic_bp_6_months")], builtin=True)


# ---------------------------------------------------------------------------
# Synthetic data
# ---------------------------------------------------------------------------
def _poisson(rng: random.Random, lam: float) -> int:
    limit, k, p = math.exp(-lam), 0, 1.0
    while True:
        p *= rng.random()
        if p <= limit:
            return k
        k += 1


def _fmt(x: float, dec: int) -> str:
    return str(int(round(x))) if dec == 0 else f"{x:.{dec}f}"


def _series_dates(unit: str, start: tuple[int, int], n: int) -> list[str]:
    y, m = start
    out = []
    for i in range(n):
        if unit == "month":
            yy, mm = y + (m - 1 + i) // 12, (m - 1 + i) % 12 + 1
            out.append(f"{yy}-{mm:02d}")
        elif unit == "quarter":
            q = (m - 1 + i)
            out.append(f"{y + q // 4}-Q{q % 4 + 1}")
        else:  # week: ISO-like dates, 7 days apart from 1 Jan of the start year
            import datetime as _dt
            d = _dt.date(y, 1, 2) + _dt.timedelta(days=7 * i)
            out.append(d.isoformat())
    return out


def generate_rows(sc: Scenario, rng: random.Random, issues: bool = False) -> tuple[list[str], list[list[str]]]:
    """Synthesize a CSV-like table whose column types QuantAI infers as intended."""
    n = rng.randint(*sc.n)
    headers = [v.header for v in sc.variables]
    cols: list[list[str]] = []
    if sc.series:
        for v in sc.variables:
            if v.kind == "date":
                cols.append(_series_dates(v.params["unit"], v.params["start"], n))
            else:
                period = {"month": 12, "week": 52, "quarter": 4}.get(sc.variables[0].params["unit"], 12)
                base, amp = v.params["base"], v.params["amp"]
                trend = rng.uniform(-0.002, 0.004) * base
                cols.append([str(max(0, int(base + trend * i + amp * math.sin(2 * math.pi * i / period) + rng.gauss(0, amp * 0.25)))) for i in range(n)])
        return headers, [list(r) for r in zip(*cols)]

    for v in sc.variables:
        p, col = v.params, []
        miss = rng.choice([0, 0, 0.01, 0.03]) if v.kind not in {"clus", "expo", "time", "event"} else 0
        for _ in range(n):
            if miss and rng.random() < miss:
                col.append("")
                continue
            if v.kind == "age":
                col.append(str(rng.randint(p["lo"], p["hi"])))
            elif v.kind == "cont":
                x = rng.gauss(p["mean"], p["sd"])
                x = max(p.get("lo", -1e9), min(p.get("hi", 1e9), x))
                col.append(_fmt(x, p.get("dec", 1)))
            elif v.kind == "count":
                col.append(str(_poisson(rng, p.get("lam", 4) * rng.uniform(0.6, 1.4))))
            elif v.kind == "bin":
                col.append(p["levels"][1] if rng.random() < p.get("p", 0.5) else p["levels"][0])
            elif v.kind in {"cat", "ord"}:
                probs = p.get("probs")
                col.append(rng.choices(p["levels"], weights=probs)[0] if probs else rng.choice(p["levels"]))
            elif v.kind == "expo":
                col.append(f"{rng.uniform(p['lo'], p['hi']):.1f}")
            elif v.kind == "time":
                col.append(f"{max(0.5, rng.expovariate(1 / p['scale'])):.1f}")
            elif v.kind == "event":
                col.append("1" if rng.random() < p.get("p", 0.4) else "0")
            elif v.kind == "clus":
                col.append(rng.choice(clusters(v.header.split(" (")[0], p["k"])))
            else:
                raise ValueError(v.kind)
        cols.append(col)
    rows = [list(r) for r in zip(*cols)]
    if issues:
        # A few classic data problems so QuantAI's data check has something to report.
        for v_index, v in enumerate(sc.variables):
            if v.kind == "age":
                for r in rng.sample(range(n), 2):
                    rows[r][v_index] = "999"
            if v.kind == "count":
                rows[rng.randrange(n)][v_index] = "99"
                break
        for r in rng.sample(range(n), 3):
            rows.append(list(rows[r]))
    return headers, rows


SCENARIOS = {sc.sid: sc for sc in TRAIN + EVAL}

# QuantAI types each label assumes for a variable kind (checked on every rendered row).
EXPECTED_TYPES = {"age": {"continuous"}, "cont": {"continuous"}, "count": {"count"}, "bin": {"binary"}, "cat": {"categorical"},
                  "ord": {"ordinal"}, "expo": {"continuous"}, "time": {"continuous"}, "event": {"binary"}, "clus": {"categorical"},
                  "date": {"id", "categorical", "continuous"}, "series": {"count", "continuous"}}
