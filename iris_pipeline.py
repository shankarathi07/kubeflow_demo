import kfp
from kfp import dsl

@dsl.component(
    base_image='python:3.9',
    packages_to_install=['scikit-learn', 'pandas']
)
def load_and_split_data_op(
    X_train_path: dsl.OutputPath(),
    X_test_path: dsl.OutputPath(),
    y_train_path: dsl.OutputPath(),
    y_test_path: dsl.OutputPath(),
):
    import pandas as pd
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split

    #load iris dataset
    iris = load_iris()

    #split data using 20% of dataset and ensure consistency with test and training data
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data, iris.target, test_size=0.2, random_state=42
    )

    pd.DataFrame(X_train).to_csv(X_train_path, index=False)
    pd.DataFrame(X_test).to_csv(X_test_path, index=False)
    pd.DataFrame(y_train).to_csv(y_train_path, index=False)
    pd.DataFrame(y_test).to_csv(y_test_path, index=False)

    print("✅ Data loaded and split successfully.")

@dsl.component(
    base_image='python:3.9',
    packages_to_install=['scikit-learn', 'pandas']
)
def train_and_evaluate_op(
    X_train_path: dsl.InputPath(),
    X_test_path: dsl.InputPath(),
    y_train_path: dsl.InputPath(),
    y_test_path: dsl.InputPath(),
):
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier

    X_train = pd.read_csv(X_train_path)
    y_train = pd.read_csv(y_train_path)
    X_test = pd.read_csv(X_test_path)
    y_test = pd.read_csv(y_test_path)

    #creates a Random Forest classifier, trains it on the training data, and then calculates its accuracy on the test data
    model = RandomForestClassifier()
    model.fit(X_train, y_train.values.ravel())
    acc = model.score(X_test, y_test.values.ravel())
    print(f"✅ Model Accuracy: {acc}")

@dsl.pipeline(
    name="Simple Iris Pipeline v2 final fix",
    description="Kubeflow Pipelines v2 with proper artifact passing"
)
def iris_pipeline():
    split = load_and_split_data_op()
    train_and_evaluate_op(
        X_train_path=split.outputs['X_train_path'],
        X_test_path=split.outputs['X_test_path'],
        y_train_path=split.outputs['y_train_path'],
        y_test_path=split.outputs['y_test_path']
    )

#create yaml file for Kubeflow 
if __name__ == '__main__':
    kfp.compiler.Compiler().compile(
        pipeline_func=iris_pipeline,
        package_path='iris_pipeline_v2_final.yaml'
    )

