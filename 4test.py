# Программа Python для иллюстрации
# соответствия шаблона

import cv2
import sys
import numpy as np

# Проверка, что аргумент (шаблон) был передан
if len(sys.argv) < 2:
    print("have no argument!    (*.png, *.jpg, ...)")
    exit()

# Работаем с камерой
img = cv2.VideoCapture(0) # Открываем камеру
img.set(cv2.CAP_PROP_FPS, 30) # Частота кадров
img.set(cv2.CAP_PROP_FRAME_WIDTH, 1280) # Ширина кадров в видеопотоке.
img.set(cv2.CAP_PROP_FRAME_HEIGHT, 720) # Высота кадров в видеопотоке.

img_rgb = None
# Список для усредненых точек найденных шаблонов на картинке(чтобы не было наслоения найденного одного и того же изображения)
# чтобы можно было проверять на уникальность
average_points = []
# Аналогично, только хранятся расстояния average_sizes[0] соответвует average_pionts[0] | [1] == [1] | ...
average_sizes = []
# Константа, хранящая допустимую разницу размера, при котором не будет записываться новое значение
# Пример: (точка 100, 100 не будет записываться, если имеются точки: 
# (100-DIFFERENCE, 100), (100, 100-DIFFERENCE), (100-DIFFERENCE, 100-DIFFERENCE), (100+DIFFERENCE, 100), ...) 
DIFFERENCE = 20
# Минимальный процент от размера шаблона
MIN = 20
# Максимальный процент от размера шаблона (максимум - 100)
MAX = 100
# Модуль разницы размеров шаблона
MODULE = 5
# Ядро свертки для размытия изображения (должно быть нечетным)
BLUR_CORE_CONVULTION = (3, 3)
# Имя файла для результатов
FILE = "result.txt"
# Порог совпадений
TRESHOLD = 0.8
# Считываем шаблон
template__ = cv2.imread(sys.argv[1] , 0)

# Поиск среднего значения из найденных точек изображений шаблона (при одинаковом размере шаблона),
# чтобы получить точно 1 значение
def findAverage(array):
    x = 0
    x_sum = 0
    y = 0
    y_sum = 0
    
    for p in array:
        x_sum += p[0]
        y_sum += p[1]

    return ((int)(x_sum/len(array)), (int)(y_sum/len(array)))

# Проверка на то, что точка еще в пределах 1го найденного изображения (из многих)
def checkPixel(minPoint, point):
    
    if point[0] > (minPoint[0] + DIFFERENCE):
        return False
    if point[1] > (minPoint[1] + DIFFERENCE):
        return False
    return True

# Добавление точки в список найденных значений, если оно не в пределах другого изображения
def addAveragePoint(points, minDif, w, h):
    # Нахоим среднее из положений одного и того же изображения
    point = findAverage(points)
    for pt in average_points:
        if point[0] <= pt[0] + minDif and point[0] >= pt[0]:
            return
        if point[0] >= pt[0] - minDif and point[0] <= pt[0]:
            return
        if point[1] <= pt[1] + minDif and point[1] >= pt[1]:
            return
        if point[1] >= pt[1] - minDif and point[1] <= pt[1]:
            return
    # Если проверка прошла, то добавляем элемент (полагая, что это новый)
    average_points.append(point)
    average_sizes.append((w,h))

def writeToFILE(data):
    # Запись результатов в файл
    file = open(FILE, "w")
    file.write(data)
    file.close()

# Создаём список координат найденых шаблонов для записи в файл
def getDataForFILE():
    i = 0
    string = ""
    while i < len(average_sizes):
        string += str(average_points[i]) + ", " + str(average_sizes[i][0]) + ", " + str(average_sizes[i][1]) + "\n"
        i += 1
    return string

# Основная функция (вызывает другие), тут проходит уменьшение шаблона и поиск шаблона на изображении
def findingWithResize(MinPercent, maxPercent, module):
    if ((MinPercent < 0 or MinPercent > 100) or (maxPercent > 100 or maxPercent < 0) or (maxPercent < MinPercent)):
        print("Wrong MIN, MAX sizes")
        exit()
    for size in range(MinPercent, maxPercent):
        # Ищем только если шаблон увеличится на module процентов
        if size%module == 0:
            # Нахождение ширины и высоты
            width = int(template__.shape[1] * size / 100)
            height = int(template__.shape[0] * size / 100)
            dsize = (width, height)
            if getContourse(cv2.resize(template__, dsize)) == True:
                return True
    return False
# Поиск шаблона
def getContourse(template):
  
    # Сохраняем ширину и высоту шаблона
    w, h = template.shape[::-1]
  
    # Выполняем операции сопоставления.
    res = cv2.matchTemplate(img_gray,template,cv2.TM_CCOEFF_NORMED)
  
    # Сохраняем координаты совпадающей области в массиве
    # Координаты упорядочены по x, y (запись проходит вдоль икса)
    loc = np.where( res >= TRESHOLD) 
    # Флаг о том, что надо прекращать (для отладки)
    flag = False

    # Инициализация флага о входе в цикл
    flag_about_in = False
    # Инициализация списка для одного изображения
    points = []
    # Инициализация минимума точки, при которой мы будем проверять, полагая, что это одно изображение
    minimum = (999999, 99999)

    for pt in zip(*loc[::-1]):
        
        if minimum[0] > pt[0]:
            if minimum[1] > pt[1]:
                minimum = pt
        # Если точка является тем же изображением, то добавляем ее в список для дальнейшего поиска среднего значения
        if checkPixel(minimum, pt) == True:
            points.append(pt)
        # В ином случае, это другое изображение
        else:
            # Добавляем старое в массив изображений
            addAveragePoint(points, DIFFERENCE, w, h)
            # Чистим список для дальнейшей записи точек нового изображения
            points.clear()
            # Устанавливаем точку как минимум для нового изобраения
            minimum = pt
            # Добавляем ее в список для дальнейшего поиска среднего значения
            points.append(pt)
        flag_about_in = True
        
        # flag = True
    
    if flag_about_in == True:    
        addAveragePoint(points, DIFFERENCE, w, h)
    return flag
# Получаем кадр с камеры
_, img_rgb = img.read()
# Добавляем размытие
img_rgb = cv2.GaussianBlur(img_rgb, BLUR_CORE_CONVULTION , 0)
# Преобразуем изображение в оттенки серого
img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
# Находим совпадения
findingWithResize(MIN, MAX, MODULE)
# Запись в файл
writeToFILE(getDataForFILE())
# Освобождаем камеру
img.release()