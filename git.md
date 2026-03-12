# Useful git ressources

## Add a branch created on remote

1. run `git fetch`
2. run `git checkout -t origin/branch-name`

## Create a remote for local repo

1. create a remote repo with the same name
2. run `git remote add origin git@github.com/gitlab.com:username/repo-name.git`
3. remember to fill both `username` and  `repo-name`, and to delete github or gitlab accordingly.
4. run `git push --set-upstream origin main`

## Delete a local branch that has been deleted from the remote

1. run `git branch -d branch-name` to delete the branch locally
2. run `git fetch --prune` to delete the reference to the remote branch

## Create a local branch and add it to the remote

1. run `git checkout -b branch-name`
2. run `git push -u origin branch-name`
