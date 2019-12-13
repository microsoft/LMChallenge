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

### Configure gpg for signing the release

- Install [gpg](https://gnupg.org) (e.g. `brew install gpg` on macOS)
- Obtain the private key for the release and its password (ask around)
- Import the key into your keyring:
  ```bash
    gpg --allow-secret-key-import --import private.key
  ```
- Ensure the key has been imported:
  ```bash
    gpg --list-keys
  ```
  Expected output:
  ```
    pub   rsa2048 2017-10-06 [SC]
          EA7AC0CCA097C391C7AA61F109F7AFCBCB48AC15
    uid           [ unknown] SwiftKey DL&NLP <swiftkey-deep@service.microsoft.com>
    sub   rsa2048 2017-10-06 [E]
  ```

  The key fingerprint is `EA7A C0CC A097 C391 C7AA 61F1 09F7 AFCB CB48 AC15`

### Configure PyPi access

In your `~/.pypirc` specify:

```
[distutils]
index-servers =
    pypi
    testpypi

[pypi]
repository: https://upload.pypi.org/legacy/
username: USERNAME
password: PASSWORD

[testpypi]
repository: https://test.pypi.org/legacy/
username: USERNAME
password: PASSWORD
```

Install `twine` for the current user

```bash
python3 -m pip install --user twine
```

### Publish a new release

 1. (optionally) update requirements `./scripts/run -i base build --no-cache && ./scripts/run -i base refreeze`
 2. run the pre-publish checks `./scripts/run check`
 3. check that you're happy with `version.txt`
 4. `rm -rf dist || true` to cleanup previously created artefacts
 5. `python3 setup.py sdist` to package a new release
 6. `twine upload --sign --identity "swiftkey-deep@" -r testpypi dist/*` to upload the release to TEST PyPi server
 7. Check the new release on test.pypi.org
 8. You can download release files and verify the signature via 
    ```
    gpg --verify lmchallenge-$(cat version.txt).tar.gz.asc lmchallenge-$(cat version.txt).tar.gz
    ```
 9. `twine upload --sign --identity "swiftkey-deep@" -r pypi dist/*` to upload the release to MAIN PyPi server at https://pypi.org
 9. `git push origin HEAD:refs/tags/$(cat version.txt)` to push the new release tag to Github
 10. Go to Github and create a release for the tag you've just pushed
 11. update, commit & push `version.txt` to start a new version
