from flask import Flask, render_template, request
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)

def calculate_deviance(academic_performance, school_behavior, family_situation):

    # Ограничение входных данных в диапазоне от 0 до 100
    academic_performance = max(0, min(academic_performance, 100))
    school_behavior = max(0, min(school_behavior, 100))
    family_situation = max(0, min(family_situation, 100))

    # Входные переменные
    academic = ctrl.Antecedent(np.arange(0, 101, 1), 'academic_performance')
    behavior = ctrl.Antecedent(np.arange(0, 101, 1), 'school_behavior')
    family = ctrl.Antecedent(np.arange(0, 101, 1), 'family_situation')

    # Выходная переменная
    deviance = ctrl.Consequent(np.arange(0, 101, 1), 'deviance_level')

    # Функции принадлежности
    academic['low'] = fuzz.trimf(academic.universe, [0, 0, 50])
    academic['medium'] = fuzz.trimf(academic.universe, [25, 50, 75])
    academic['high'] = fuzz.trimf(academic.universe, [50, 100, 100])

    behavior['poor'] = fuzz.trimf(behavior.universe, [0, 0, 50])
    behavior['average'] = fuzz.trimf(behavior.universe, [25, 50, 75])
    behavior['good'] = fuzz.trimf(behavior.universe, [50, 100, 100])

    family['unstable'] = fuzz.trimf(family.universe, [0, 0, 50])
    family['average'] = fuzz.trimf(family.universe, [25, 50, 75])
    family['stable'] = fuzz.trimf(family.universe, [50, 100, 100])

    deviance['low'] = fuzz.trimf(deviance.universe, [0, 0, 50])
    deviance['medium'] = fuzz.trimf(deviance.universe, [25, 50, 75])
    deviance['high'] = fuzz.trimf(deviance.universe, [50, 100, 100])

    # Правила
    rule1 = ctrl.Rule(academic['low'] & behavior['poor'] & family['unstable'], deviance['high'])
    rule2 = ctrl.Rule(academic['low'] & behavior['poor'] & family['average'], deviance['medium'])
    rule3 = ctrl.Rule(academic['medium'] & behavior['average'] & family['stable'], deviance['low'])
    rule4 = ctrl.Rule(academic['high'] & behavior['good'] & family['stable'], deviance['low'])
    rule5 = ctrl.Rule(academic['medium'] & behavior['poor'] & family['unstable'], deviance['high'])
    rule6 = ctrl.Rule(academic['high'] & behavior['average'] & family['unstable'], deviance['medium'])
    rule7 = ctrl.Rule(academic['low'] & behavior['good'] & family['average'], deviance['medium'])
    rule8 = ctrl.Rule(academic['medium'] & behavior['average'] & family['unstable'], deviance['high'])
    rule9 = ctrl.Rule(academic['medium'] & behavior['poor'] & family['average'], deviance['medium'])
    rule10 = ctrl.Rule(academic['low'] & behavior['poor'] & family['stable'], deviance['high'])
    rule11 = ctrl.Rule(academic['high'] & behavior['good'] & family['unstable'], deviance['medium'])
    rule12 = ctrl.Rule(academic['medium'] & behavior['good'] & family['stable'], deviance['low'])

    # Создание системы управления
    deviance_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9, rule10, rule11, rule12])
    deviance_simulation = ctrl.ControlSystemSimulation(deviance_ctrl)

    # Ввод данных
    deviance_simulation.input['academic_performance'] = academic_performance
    deviance_simulation.input['school_behavior'] = school_behavior
    deviance_simulation.input['family_situation'] = family_situation

    # Вычисление
    deviance_simulation.compute()

    return deviance_simulation.output['deviance_level']

# Ограничение входных данных в диапазоне от 0 до 100
    academic_performance = max(0, min(academic_performance, 100))
    school_behavior = max(0, min(school_behavior, 100))
    family_situation = max(0, min(family_situation, 100))

def plot_membership_functions():
    # Определение области значений
    x_academic = np.arange(0, 101, 1)
    x_behavior = np.arange(0, 101, 1)
    x_family = np.arange(0, 101, 1)
    x_deviance = np.arange(0, 101, 1)

    # Функции принадлежности
    academic_low = fuzz.trimf(x_academic, [0, 0, 50])
    academic_medium = fuzz.trimf(x_academic, [25, 50, 75])
    academic_high = fuzz.trimf(x_academic, [50, 100, 100])

    behavior_poor = fuzz.trimf(x_behavior, [0, 0, 50])
    behavior_average = fuzz.trimf(x_behavior, [25, 50, 75])
    behavior_good = fuzz.trimf(x_behavior, [50, 100, 100])

    family_unstable = fuzz.trimf(x_family, [0, 0, 50])
    family_average = fuzz.trimf(x_family, [25, 50, 75])
    family_stable = fuzz.trimf(x_family, [50, 100, 100])

    deviance_low = fuzz.trimf(x_deviance, [0, 0, 50])
    deviance_medium = fuzz.trimf(x_deviance, [25, 50, 75])
    deviance_high = fuzz.trimf(x_deviance, [50, 100, 100])

    # Построение графиков
    fig, axs = plt.subplots(4, 1, figsize=(10, 10))

    axs[0].plot(x_academic, academic_low, 'b', label='Low')
    axs[0].plot(x_academic, academic_medium, 'g', label='Medium')
    axs[0].plot(x_academic, academic_high, 'r', label='High')
    axs[0].set_title('Academic Performance')
    axs[0].legend()

    axs[1].plot(x_behavior, behavior_poor, 'b', label='Poor')
    axs[1].plot(x_behavior, behavior_average, 'g', label='Average')
    axs[1].plot(x_behavior, behavior_good, 'r', label='Good')
    axs[1].set_title('School Behavior')
    axs[1].legend()

    axs[2].plot(x_family, family_unstable, 'b', label='Unstable')
    axs[2].plot(x_family, family_average, 'g', label='Average')
    axs[2].plot(x_family, family_stable, 'r', label='Stable')
    axs[2].set_title('Family Situation')
    axs[2].legend()

    axs[3].plot(x_deviance, deviance_low, 'b', label='Low')
    axs[3].plot(x_deviance, deviance_medium, 'g', label='Medium')
    axs[3].plot(x_deviance, deviance_high, 'r', label='High')
    axs[3].set_title('Deviance Level')
    axs[3].legend()

    plt.tight_layout()
    plt.show()

# Вызываем функцию для построения графиков функций принадлежности
plot_membership_functions()

# Маршрут для обработки AJAX-запросов
@app.route('/predict', methods=['POST'])
def predict():
    academic_performance = float(request.form['academic_performance'])
    school_behavior = float(request.form['school_behavior'])
    family_situation = float(request.form['family_situation'])

    deviance_level = calculate_deviance(academic_performance, school_behavior, family_situation)

    return str(deviance_level)

# Маршрут для отображения HTML-шаблона
@app.route('/')
def index():

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
