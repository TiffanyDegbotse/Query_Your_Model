@echo off
echo Building Iris reasoning index...

python Query_Your_Model/scripts/build_base_index.py ^
  --model_path Query_Your_Model/model_data/model.pkl ^
  --csv Query_Your_Model/model_data/data.csv ^
  --features "sepal length (cm),sepal width (cm),petal length (cm),petal width (cm)" ^
  --target target ^
  --namespace Query_Your_Model/data/base_indices/iris_global ^
  --sample 100

echo Done! Index saved to Query_Your_Model/data/base_indices/iris_global
pause
