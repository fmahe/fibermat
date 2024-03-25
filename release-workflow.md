# Release Workflow Guide

1. Review `jupyter-notebook.ipynb` in JupyterLab.

    - Ensure README content is accurate (title should be "Welcome to FiberMat’s tutorial!").

    - Verify installed package versions.

    - Check outputs and collapse all code cells.

    - Save `jupyter-notebook.ipynb`.

2. Verify Package Metadata in `fibermat.__init__.py`.

    - Confirm or update the version specified in `fibermat.__version__`.

    - Check for accurate copyright information.

3. Review `README.md`.

    - Test links to ensure they're functional.

    - Check for accurate copyright.

    - Review citation references.

4. Update Version in `pyproject.toml`.

    - Update the version number in the `pyproject.toml` file.

5. Generate Sources.

    - Run the following command to generate sources:
    ```shell
    ./make --all

    ```

6. Commit and Push Changes.

    - Commit and push the changes to the GitHub repository.

7. Release Package on PyPi.

    - Use the following command to upload the distribution files (see [PyPi Guide](https://packaging.python.org/en/latest/tutorials/packaging-projects/)):
    ```shell
    twine upload dist/*

    ```
