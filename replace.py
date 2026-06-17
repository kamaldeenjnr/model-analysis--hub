f_index = 'c:/Users/amadu/Desktop/model-analysis-hub/new_site/index.html'
f_founder = 'c:/Users/amadu/Desktop/model-analysis-hub/new_site/founder.html'
f_projects = 'c:/Users/amadu/Desktop/model-analysis-hub/new_site/projects.html'
f_ai = 'c:/Users/amadu/Desktop/model-analysis-hub/new_site/ai.html'
f_llms = 'c:/Users/amadu/Desktop/model-analysis-hub/new_site/llms.txt'
f_js = 'c:/Users/amadu/Desktop/model-analysis-hub/new_site/quantai.js'

def replace_in_file(path, old, new):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()
        if old in content:
            content = content.replace(old, new)
            with open(path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f'Replaced in {path}')
    except Exception as e:
        print(f'Error in {path}: {e}')

replace_in_file(f_index, 'Building robust machine learning pipelines and forecasting models', 'Building robust statistical forecasting models')
replace_in_file(f_index, 'AI & Predictive Modeling', 'Predictive Modeling')
replace_in_file(f_index, 'Machine Learning', 'Statistical Modeling')

replace_in_file(f_founder, 'QGIS', 'ARC GIS')
replace_in_file(f_founder, '<span class=\"skill-pill\"><i class=\"fas fa-database text-purple-600\"></i> SQL / MySQL</span>', '<span class=\"skill-pill\"><i class=\"fas fa-database text-purple-600\"></i> SQL</span>')
replace_in_file(f_founder, '<span class=\"skill-pill\"><i class=\"fas fa-brain text-indigo-600\"></i> Machine Learning</span>', '')
replace_in_file(f_founder, '<span class=\"skill-pill\"><i class=\"fas fa-globe text-blue-500\"></i> REST APIs</span>', '')
replace_in_file(f_founder, 'Data Science & AI', 'Data Science & Analytics')

replace_in_file(f_projects, 'QGIS', 'ARC GIS')
proj_old = 'A full-stack project management and data collection platform built for scalable team coordination. Includes secure authentication, form-based data gathering, and a responsive admin dashboard.'
proj_new = 'This is a comprehensive, full-stack student project management system designed for university-level research work. It supports Undergraduate, Master\'s, and PhD level projects with a complete workflow from draft submission to final publication, this is projectflow and it was completed 2025.'
replace_in_file(f_projects, proj_old, proj_new)
replace_in_file(f_projects, 'HTML, CSS, JS, PHP, MySQL', 'HTML, CSS, JS, SQL')
replace_in_file(f_projects, 'Probabilistic system calculating', 'Mathematical system calculating')
replace_in_file(f_projects, 'Exceedance Probability Analytics', 'Risk Modeling Analytics')

replace_in_file(f_ai, 'REST architecture', 'integration architecture')

replace_in_file(f_llms, 'QGIS', 'ARC GIS')
replace_in_file(f_llms, 'MySQL', 'SQL')
replace_in_file(f_llms, 'PHP', '')
replace_in_file(f_llms, 'Machine learning and AI system development', 'Statistical modeling and analytics')
replace_in_file(f_llms, 'REST API', 'API')
replace_in_file(f_llms, 'Probabilistic modeling and exceedance probability analytics', 'Mathematical risk modeling and analytics')
replace_in_file(f_llms, 'Exceedance Probability Analytics', 'Risk Modeling Analytics')
replace_in_file(f_llms, 'Probabilistic system calculating', 'Mathematical system calculating')
replace_in_file(f_llms, 'Full-stack project management and data collection platform with secure auth and admin dashboard.', proj_new)

replace_in_file(f_js, 'Probabilistic models', 'Mathematical models')
replace_in_file(f_js, 'Exceedance Probability', 'Risk Modeling')
