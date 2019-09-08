workflow "Dummy" {
  resolves = ["dummy"]
  on = "push"
}

action "dummy" {
  uses = "docker://cdssnc/touched-github-action"
  args = "{*}"
}
