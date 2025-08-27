import kfp
from kfp import dsl

# Define constants for the pipeline
TEST_SIZE = 0.2
RANDOM_STATE = 42

@dsl.component(
    base_image='python:3.9',
    packages_to_install=['scikit-learn', 'pandas']
)
def load_and_split_data_op(
    x_train_path: dsl.OutputPath(),
    x_test_path: dsl.OutputPath(),
    y_train_path: dsl.OutputPath(),
    y_test_path: dsl.OutputPath(),
):
    """
    Loads the Iris dataset, splits it into training and testing sets,
    and saves them as CSV files.
    """
    import pandas as pd
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split

    iris = load_iris()
    x_train, x_test, y_train, y_test = train_test_split(
        iris.data, iris.target, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    pd.DataFrame(x_train).to_csv(x_train_path, index=False)
    pd.DataFrame(x_test).to_csv(x_test_path, index=False)
    pd.DataFrame(y_train).to_csv(y_train_path, index=False)
    pd.DataFrame(y_test).to_csv(y_test_path, index=False)

    print("✅ Data loaded and split successfully.")

@dsl.component(
    base_image='python:3.9',
    packages_to_install=['scikit-learn', 'pandas']
)
def train_and_evaluate_op(
    x_train_path: dsl.InputPath(),
    x_test_path: dsl.InputPath(),
    y_train_path: dsl.InputPath(),
    y_test_path: dsl.InputPath(),
):
    """
    Trains a RandomForestClassifier on the training data and evaluates it on the test data.
    """
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier

    x_train = pd.read_csv(x_train_path)
    y_train = pd.read_csv(y_train_path)
    x_test = pd.read_csv(x_test_path)
    y_test = pd.read_csv(y_test_path)

    # Initialize and train the model
    model = RandomForestClassifier()
    model.fit(x_train, y_train.values.ravel())

    # Evaluate the model
    accuracy = model.score(x_test, y_test.values.ravel())
    print(f"✅ Model Accuracy: {accuracy}")

@dsl.pipeline(
    name="Iris Classification Pipeline",
    description="A simple pipeline that trains a classifier on the Iris dataset."
)
def iris_pipeline():
    """Defines the Iris classification pipeline."""
    split_task = load_and_split_data_op()
    train_and_evaluate_op(
        x_train_path=split_task.outputs['x_train_path'],
        x_test_path=split_task.outputs['x_test_path'],
        y_train_path=split_task.outputs['y_train_path'],
        y_test_path=split_task.outputs['y_test_path']
    )

if __name__ == '__main__':
    # Compile the pipeline to a YAML file for Kubeflow
    kfp.compiler.Compiler().compile(
        pipeline_func=iris_pipeline,
        package_path='iris_pipeline.yaml'
    )

