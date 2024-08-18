import os
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from catboost import CatBoostRegressor

def read_txt_file(file_path):
    """
    Чтение текстового файла и преобразование его содержимого в массив numpy.
    """
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    # Пропуск первых 4 строк (в зависимости от структуры данных)
    data_lines = lines[4:]
    
    # Преобразование строк в список чисел
    data = []
    for line in data_lines:
        values = list(map(float, line.strip().split()))
        data.extend(values)
    
    # Преобразование списка в numpy массив
    data_array = np.array(data)
    
    return data_array

def load_property_maps(data_folder):
    """
    Загружает и парсит все файлы в папке, возвращает словарь с массивами numpy.
    """
    property_maps = {}
    
    for file_name in os.listdir(data_folder):
        if file_name.endswith('.txt'):
            file_path = os.path.join(data_folder, file_name)
            property_name = os.path.splitext(file_name)[0]
            try:
                property_maps[property_name] = read_txt_file(file_path)
            except ValueError as e:
                print(f"Ошибка при обработке файла {file_name}: {e}")
    
    return property_maps

def obtain_data(property_maps, well_coordinates, distances):
    """
    Функция для сбора геологических данных вдоль стволов скважин.
    """
    data = []
    for coord in well_coordinates:
        x, y = coord
        
        row = []
        for distance in distances:
            # Определяем границы вокруг точки с учетом дистанции
            x_start = max(x - distance, 0)
            x_end = min(x + distance, 171)
            y_start = max(y - distance, 0)
            y_end = min(y + distance, 121)
            
            for property_name, property_map in property_maps.items():
                region_values = property_map[x_start:x_end, y_start:y_end]
                mean_value = np.mean(region_values)
                row.append(mean_value)
        
        data.append(row)
    
    return pd.DataFrame(data, columns=[f"{prop}_{dist}" for prop in property_maps.keys() for dist in distances])

def predict_production(data_df, model):
    """
    Функция для предсказания добычи с использованием модели.
    """
    predictions = model.predict(data_df)
    return predictions

def calculate_npv(oil_rate_list, bhp_list, length_list):
    """
    Функция для расчета NPV (Net Present Value).
    """
    # Подключите свою функцию расчета NPV
    return NPV_function(oil_rate_list, bhp_list, length_list)

def optimize_wells(initial_wells, property_maps, model, max_iterations=100):
    def objective(well_params):
        well_coords = well_params[:len(initial_wells)*2].reshape(-1, 2)
        bhp = well_params[len(initial_wells)*2:len(initial_wells)*3]
        lengths = well_params[len(initial_wells)*3:]
        
        data_df = obtain_data(property_maps, well_coords, [1, 5, 10])
        oil_rate_predictions = model.predict(data_df)
        
        npv_value = calculate_npv(oil_rate_predictions, bhp, lengths)
        
        return -npv_value
    
    result = minimize(objective, initial_wells, method='L-BFGS-B', options={'maxiter': max_iterations})
    optimized_params = result.x
    
    return optimized_params, -result.fun

def main_pipeline(data_folder, model):
    # Загрузка геологических карт
    property_maps = load_property_maps(data_folder)
    
    # Генерация начальных параметров для скважин
    initial_wells = generate_initial_wells()  # Функция для задания начальных координат и параметров
    
    # Оптимизация расположения и параметров скважин
    optimized_wells, optimized_npv = optimize_wells(initial_wells, property_maps, model)
    
    return optimized_wells, optimized_npv

# Загрузка модели
model = CatBoostRegressor()
model.load_model("ML_model")

# Основной пайплайн
data_folder = "data"
optimized_wells, optimized_npv = main_pipeline(data_folder, model)
print("Optimized Wells:", optimized_wells)
print("Optimized NPV:", optimized_npv)
