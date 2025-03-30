import os

scripts = [
        "Zadanie2_1.py",
        "Zadanie2_2.py",
        "Zadanie2_3.py",
        "Zadanie2_4.py",
    ]

def main():
    while True:
        print("Wybierz zadanie do uruchomienia: ")
        for i, script in enumerate(scripts, start=1):
            print(f"{i}. {script}")
        print("0. Wyjście")

        choice = input("Podaj numer: ")

        if choice == "0":
            print("Zamykanie programu...")
            break

        if choice.isdigit() and int(choice) >= 1 and int(choice) <= 5:
            os.system(f'python {scripts[int(choice) - 1]}')
        else:
            print("Niepoprawny wybór, spróbuj ponownie.")


if __name__ == "__main__":
    main()