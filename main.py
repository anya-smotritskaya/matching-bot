import gspread
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Подключение к Google Sheets
def connect_to_sheets():
    gc = gspread.service_account(filename='credentials.json')
    sh = gc.open("AIESEC_Matching_System")
    return sh

def run_matching():
    print("Запуск алгоритма подбора...")
    sh = connect_to_sheets()
    
    # 1. Загружаем листы
    raw_sheet = sh.worksheet("raw_responses")
    project_sheet = sh.worksheet("project_db")
    rec_sheet = sh.worksheet("recommendations")
    
    # 2. Превращаем данные в таблицы (Pandas)
    candidates = pd.DataFrame(raw_sheet.get_all_records())
    projects = pd.DataFrame(project_sheet.get_all_records())
    
    # 3. Смотрим, кого мы уже обработали
    existing_records = rec_sheet.get_all_records()
    processed_emails = [row['Email'] for row in existing_records]
    
    # 4. Находим новеньких
    new_candidates = candidates[~candidates['Email'].isin(processed_emails)]
    
    if new_candidates.empty:
        print("Новых кандидатов нет.")
        return

    print(f"Найдено новых кандидатов: {len(new_candidates)}. Начинаю подбор...")

    # 5. Обрабатываем каждого новенького
    for index, cand in new_candidates.iterrows():
        
        # --- Собираем текст из анкеты кандидата ---
        motivation = str(cand.get('Опишите вашу мотивацию и вклад, который вы хотите внести', ''))
        spheres = str(cand.get('В каких сферах у вас есть опыт или интересы?', ''))
        sdg = str(cand.get('Какие Цели устойчивого развития (ЦУР) вам наиболее близки?', ''))
        
        # Склеиваем в одну большую строку для анализа
        candidate_text = f"{motivation} {spheres} {sdg}"
        
        # --- Сравниваем с каждым проектом из базы ---
        similarities = []
        for p_idx, proj in projects.iterrows():
            # Собираем текст проекта из ВСЕХ информативных колонок
            project_name = str(proj.get('Название', ''))
            project_sdg = str(proj.get('Цель (SDG)', ''))
            project_skills = str(proj.get('Требуемые навыки (Теги)', ''))
            project_desc = str(proj.get('Описание', ''))
            
            # Склеиваем в один текстовый "паспорт проекта"
            project_text = f"{project_name} {project_sdg} {project_skills} {project_desc}"
            
            # Магия TF-IDF: создаем "облако смыслов" из двух текстов
            vectorizer = TfidfVectorizer()
            try:
                tfidf_matrix = vectorizer.fit_transform([candidate_text, project_text])
                # Считаем косинусное сходство (от 0 до 1)
                score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            except:
                score = 0  # Если текст пустой или ошибка - сходство 0
                
            similarities.append(score)
            
        # 6. Находим индексы ТОП-3 проектов
        sim_array = np.array(similarities)
        top_indices = sim_array.argsort()[-3:][::-1]  # Сортируем и берем последние 3
        
        # 7. Записываем результат в лист рекомендаций
        row_to_add = [cand['Email']]
        for idx in top_indices:
            proj_name = projects.iloc[idx]['Название']
            proj_country = projects.iloc[idx]['Страны (примеры)']
            row_to_add.append(f"{proj_name} ({proj_country})")
            
        # Если проектов меньше 3, добиваем пустыми строками
        while len(row_to_add) < 4:
            row_to_add.append("Не найдено")
            
        rec_sheet.append_row(row_to_add)
        print(f"Обработан: {cand['Email']}")

    print("Работа завершена.")

if __name__ == "__main__":
    run_matching()
