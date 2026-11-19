# Releasing CzechSpeech for the web

Both SDKs are released together and with the same version number: this one to npm as `@moravianai/czechspeech`, the
Android SDK (`czechspeech-android`, its own `RELEASING.md`) to Maven Central. A version published to npm cannot be
published again, and unpublishing is limited; a mistake is fixed by a new version number.

The release needs things only the maintainer has: the GitHub organisation and the npm account that owns the
`@moravianai` scope (the account `moravianai`, which also published `@moravianai/thinembed-vq`), with two-factor
authentication. Nothing here is done by CI.

## What must already be true

- The model the SDK downloads is public: `MODELS[DEFAULT_MODEL_ID]` in `src/catalog.js` points at a pinned revision
  of <https://huggingface.co/moravianai/CzechSpeech> (size and SHA-256 next to it).
- The public catalogue holds the free Czech model only. No enterprise model (the Czech, Slovak and English one) goes
  into `MODELS` in `src/catalog.js`: it is gated on Hugging Face and a licensed customer hosts the file himself and
  passes URL, SHA-256 and `languages` to `CzechSpeechModel.fromUrl()` (README, "Your own or a licensed model").
- `npm whoami` prints `moravianai` (run `npm login` first if not; a stored token that has expired gives `E401`), and
  `npm access list packages @moravianai` lists `@moravianai/thinembed-vq` as `read-write`.

## 1. Prepare the release commit

```bash
git switch main && git merge --ff-only vydani-0.3.0      # or merge the release branch the usual way
git submodule update --init
native/build-wasm.sh all          # WSL or Linux: emsdk 6.0.13 in ~/emsdk; builds the three modules
npm ci
```

- `src/wasm/` is not in git: the package ships what the build above produced. `npm pack` refuses modules that are
  missing, built without the patched WebGPU kernel, without the runtime teardown or with unfixed glue (`prepack`).
- `CHANGELOG.md` has the heading `## <version> (<date>)` with the date of the release (the release branch carries
  `## 0.3.0 (YYYY-MM-DD)`; replace the placeholder on the day, commit it, and only then pack: `prepack` and
  `check:tarball` refuse a heading without a real date, so a tarball packed before release day cannot be published).
- The version is the same in `package.json`, `package-lock.json`, `src/index.js` (`VERSION`), the first line of
  `types/index.d.ts` and the README's first paragraph.

```bash
npm run lint                      # ESLint (recommended rules), no warnings allowed
npm test                          # unit tests with coverage thresholds (lines 95, branches 86, functions 90), the public API listing included
npm run test:bundlers
npm run test:integration          # needs CZECHSPEECH_MODEL, CZECHSPEECH_TEST_AUDIO, CZECHSPEECH_TEST_MANIFEST (README)
npm run test:e2e                  # headless Chromium, CPU and WebGPU; same variables
npm run check:tarball             # packs, installs the tarball into an empty project, imports it, type-checks a consumer
git status --short                # must print nothing
```

`npm run check:tarball` leaves `moravianai-czechspeech-<version>.tgz` in the repository (ignored by git) and lists
its files: no test, script or native source may be among them.

## 2. Public history

The public repository `moravian-ai/czechspeech-web` (remote `public`) holds one squashed commit per release, authored
`Moravian AI <info@moravian.ai>`, with the tree of the release branch; the development history stays in a private
development repository (`origin`). The next release is a new commit on top of `public/main`, never a force push. Make the
squash on release day, after the date is committed, as a local branch (do not push it to `origin`):

```bash
git fetch public
git switch vydani-<version>
c=$(GIT_AUTHOR_NAME="Moravian AI" GIT_AUTHOR_EMAIL=info@moravian.ai GIT_COMMITTER_NAME="Moravian AI" GIT_COMMITTER_EMAIL=info@moravian.ai \
    git commit-tree 'HEAD^{tree}' -p public/main -F <message-file>) && git branch -f vydani-<version>-public "$c"
git diff vydani-<version> vydani-<version>-public                    # must print nothing
test "$(git rev-parse vydani-<version>-public^)" = "$(git rev-parse public/main)" || echo "WRONG PARENT"
git log -1 --format='%an <%ae> / %cn <%ce>' vydani-<version>-public  # Moravian AI <info@moravian.ai> twice
git push public vydani-<version>-public:main                          # fast-forward; never --tags, --mirror or --force
```

The repository may stay private at this point. It is made public only after `npm publish` has succeeded (step 3), right
before the tag: `package.json` (`repository`) and the README link to <https://github.com/moravian-ai/czechspeech-web>,
and a failed pack or publish that needs a change of the tree is then still fixed without a second public commit.

## 3. npm

Pack on release day, after the date is filled in (step 1), and publish exactly that tarball:

```bash
npm run check:tarball
tar -xzOf moravianai-czechspeech-<version>.tgz package/CHANGELOG.md | grep -m1 '^## <version> (20[0-9][0-9]-[0-9][0-9]-[0-9][0-9])'
tar -xzOf moravianai-czechspeech-<version>.tgz package/package.json | grep '"version": "<version>"'
npm profile get email             # the account's public e-mail: must be info@moravian.ai, not a personal one
npm publish ./moravianai-czechspeech-<version>.tgz --access public --dry-run
npm publish ./moravianai-czechspeech-<version>.tgz --access public   # asks for the one-time password
```

A tarball left over from an earlier run is never published: delete it and pack again. `npm publish` runs `prepack` again.
There is no provenance attestation: that needs a publish from GitHub Actions (trusted publishing), which can be set up for
the next version once the repository is public.

## 4. Check the published package from an empty project

```bash
mkdir czechspeech-check && cd czechspeech-check && npm init -y > /dev/null
npm install @moravianai/czechspeech@<version>
node --input-type=module -e "import { VERSION, DEFAULT_MODEL, MODELS } from '@moravianai/czechspeech'; console.log(VERSION, Object.keys(MODELS), DEFAULT_MODEL.url)"
```

Then open the example (`examples/basic`, served with `npm run serve`) or an app against the published version in
Chrome: the model downloads from Hugging Face, its SHA-256 is verified by the SDK, and a spoken sentence comes back
as a transcript, on WebGPU where the browser has it.

## 5. Tag

```bash
# the tag records the committer's name and e-mail: never the personal git identity
GIT_COMMITTER_NAME="Moravian AI" GIT_COMMITTER_EMAIL=info@moravian.ai git tag -a v<version> vydani-<version>-public -m "@moravianai/czechspeech <version>"
git cat-file -p v<version> | grep '^tagger'      # Moravian AI <info@moravian.ai>
git push public v<version>                       # one tag; never --tags
```

## What the tools do, and what they leave to you

| step | who |
|---|---|
| `prepack`: the worklet is up to date, every listed file exists, the VAD hash, the versions agree, no build-machine path, the modules are the patched and fixed ones | npm, on `pack` and `publish` |
| `check:tarball`: installs the tarball into an empty project, imports it, type-checks a consumer | you run it (step 3) |
| building the WebAssembly modules; the browser and integration tests with the model | you (step 1) |
| `git push` to `public`, `npm publish` with the one-time password, making the repository public, the tag | you |
