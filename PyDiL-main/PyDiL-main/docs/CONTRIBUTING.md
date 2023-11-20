# PyDiL Contributing Guidelines

The current contribution guidelines are inspired by the [POT Toolbox Guidelines](https://pythonot.github.io/master/contributing.html)

## How to Contribute

1. Clone the github repository to your local disk [via ssh](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/using-ssh-agent-forwarding),

```sh
$ git clone git@github.com:eddardd/PyDiL.git
```

2. Create a ```feature``` branch for your development,

```sh
$ git checkout -b my-feature
```

3. Use ```git add``` and ```git commit``` to add changes to your branch,

```sh
$ git add modified_files
$ git commit -m "some useful message"
```

After commiting, you integrate your changes to the remote version of the repository through ```git push```,

```sh
$ git push -u origin my-feature
```

4. Create a [Pull Request](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request-from-a-fork).

## Code Conventions

Here are a list of convetions that this repository follows,

- PEP8 Guidelines. For formatting your code, you can use [autopep8](https://pypi.org/project/autopep8/).
- All public methods should have informative docstrings.