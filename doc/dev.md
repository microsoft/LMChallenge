# Developing

## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.microsoft.com.

When you submit a pull request, a CLA-bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., label, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Developing & testing

We recommend developing using Docker, although virtual env is also very good (these tools are useful to make sure that you're using a stable, reproducible, environment). The Python script `./scripts/run` is our main development tool which automates build & checking tasks.

**Tests** and general checks are currently are run with:

    ./scripts/run build
    ./scripts/run check

For quicker tests, while developing, try `./scripts/run test`.

**Documentation** may be built in the `/site` folder using:

    ./scripts/run doc

## Publishing

 1. (optionally) update requirements `./scripts/run -i base build --no-cache && ./scripts/run -i base refreeze`
 2. run the pre-publish checks `./scripts/run check`
 3. check that you're happy with `version.txt`
 4. `python3 setup.py sdist upload -r pypi` (you must first set up pypi in `~/.pypirc`, as below & provide GPG credentials with `gpg --import`)
    ```
    [distutils]
    index-servers =
        pypi

    [pypi]
    repository: https://upload.pypi.org/legacy/
    username: USERNAME
    password: PASSWORD
    ```
 5. `git push origin HEAD:refs/tags/$(cat version.txt)`
 6. update, commit & push `version.txt`
