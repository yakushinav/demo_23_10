# Пример 2. Нечеткие множества
def input_int(message, error_message="Ошибка"):
    raw = ""
    while not isinstance(raw, (int, )):
        raw = input(message+"\n")
        try:
            raw = int(raw)
        except ValueError:
            print(error_message)
    return raw

def input_float(message, error_message="Ошибка"):
    raw = ""
    while not isinstance(raw, (float, )):
        raw = input(message+"\n")
        try:
            raw = float(raw)
        except ValueError:
            print(error_message)
    return raw


MAX_PRECENTAGE = 18.0
MIN_PRECENTAGE = 6.25
MIN_YEAR = 12
MAX_YEAR = 12*20

BEST_PRECENTAGE_BOARDER = 12.0
MIDDLE_PRECENTAGE_BOARDER = 15.0
BEST_YEAR_BORDER = 24
MIDDLE_YEAR_BORDER = 70


def precentage_good(x):
    if x > BEST_PRECENTAGE_BOARDER:
        return 0
    elif x < MIN_PRECENTAGE:
        return 1
    return 1 - abs(MIN_PRECENTAGE-x)/(BEST_PRECENTAGE_BOARDER-MIN_PRECENTAGE)

def precentage_middle(x):
    if (x < BEST_PRECENTAGE_BOARDER) or (x > MIDDLE_PRECENTAGE_BOARDER):
        return 0
    return 1 - abs(MAX_PRECENTAGE-x)/(MAX_PRECENTAGE-MIN_PRECENTAGE)

def precentage_bad(x):
    if (x < MIDDLE_PRECENTAGE_BOARDER):
        return 0
    if (x > MAX_PRECENTAGE):
        return 1
    return 1-(MAX_PRECENTAGE-x)/(MAX_PRECENTAGE-MIDDLE_PRECENTAGE_BOARDER)


def year_good(x):
    if (x > BEST_YEAR_BORDER):
        return 0
    elif (x < MIN_YEAR):
        return 1
    return 1 - abs(MIN_YEAR-x)/(BEST_YEAR_BORDER-MIN_YEAR)

def year_middle(x):
    if (x > MIDDLE_YEAR_BORDER) or (x < BEST_YEAR_BORDER):
        return 0
    return 1 - (MIDDLE_YEAR_BORDER-x)/(MAX_YEAR-MIN_YEAR)

def year_bad(x):
    if (x < MIDDLE_YEAR_BORDER):
        return 0
    if (x > MAX_YEAR):
        return 1
    return 1 - (MAX_YEAR - x)/(MAX_YEAR-MIDDLE_YEAR_BORDER)


input_precentage = input_float("Введите % ставку ")
input_time = input_int("Введите срок (месяцев) ")

take_degree = ((precentage_good(input_precentage) + year_good(input_time)) - precentage_bad(input_precentage) - year_bad(input_time) + 0.5*(year_middle(input_time)+precentage_middle(input_precentage)))/2
disagree_degree = (precentage_bad(input_precentage) + year_bad(input_time) - precentage_good(input_precentage) - year_good(input_time) + 0.5*(year_middle(input_time)+precentage_middle(input_precentage)))/2

print(f"Процентая ставка: хор:{precentage_good(input_precentage)} сред:{precentage_middle(input_precentage)} плохо:{precentage_bad(input_precentage)}")
print(f"Продолжительность кредита: хор:{year_good(input_time)} сред:{year_middle(input_time)} плохо:{year_bad(input_time)}")
#print(take_degree, disagree_degree)
if take_degree > disagree_degree:
    print("Хорошие условия")
else:
    print("Плохие условия")