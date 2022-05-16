import json


class JsonReader:
    def __init__(self, path):
        with open(path, "r") as raw_json:
            self.data = json.load(raw_json, strict=False)
        self.excel_path = self.data["csv_location"]
        self.x_column_name = self.data["x_column_name"]
        self.y_column_name = self.data["y_column_name"]
        self.x_error_column_name = self.data["x_error_column_name"]
        self.y_error_column_name = self.data["y_error_column_name"]
        self.x_label = self.data["x_label"]
        self.y_label = self.data["y_label"]
        self.graph_title = self.data["graph_title"]
        self.fit_equation = self.data["fit_equation"]

    def get_equation(self):
        return self.fit_equation

    def get_parameters(self):
        return (self.excel_path,
                self.x_column_name,
                self.y_column_name,
                self.x_error_column_name,
                self.y_error_column_name,
                self.x_label,
                self.y_label,
                self.graph_title,
                self.fit_equation)
