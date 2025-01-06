def print_features_info(features):
    print("------------------------------------")
    print(f"num registros: {features.shape[0]}")
    print(f"num atributos: {features.shape[1]}")
    print(f"atributos: {features.columns}")
    print("------------------------------------")


def format_duration(before, after):
    return f"{((after-before)*1000):.2f}"
