Three independent configuration files sit in the current directory: `alpha.json`,
`beta.json`, `gamma.json`. They are unrelated. Edit all three; each change must be
made without affecting the others.

1. `alpha.json`: set `active` to `true` and `retries` to `7`. Keep `name`.
2. `beta.json`: append the string `"y"` to the `tags` array (so it becomes
   `["x", "y"]`) and set `priority` to `1`. Keep `name`.
3. `gamma.json`: under `limits`, set `mem` to `1024` and add `gpu` set to `"T4"`.
   Keep `limits.cpu` and `name`.

Do not create extra files and do not edit any file other than these three. Write
the edits into the files themselves.
