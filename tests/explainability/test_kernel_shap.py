from chap_core.explainability.kernel_shap import explain_shap
from chap_core.file_io.example_data_set import datasets


def test_explain_shap():
    dataset = datasets["ISIMIP_dengue_harmonized"].load()["brazil"]
    location = list(dataset.locations())[0]
    #TODO bug when horizon=1, lags are missing
    explain_shap(model=None, dataset=dataset, location=location, horizon=3)
