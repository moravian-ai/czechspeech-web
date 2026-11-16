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
- `npm whoami` prints `moravianai` (run `npm login` first if not; a stored token that has expired gives `E401`), and
  `npm access list packages @moravianai` lists `@moravianai/thinembed-vq` as `read-write`.

## 1. Prepare the release commit

```bash
git switch main && git merge --ff-only vydani-0.2.0      # or merge the release branch the usual way
git submodule update --init
native/build-wasm.sh all          # WSL or Linux: emsdk 6.0.13 in ~/emsdk; builds the three modules
npm ci
```

- `src/wasm/` is not in git: the package ships what the build above produced. `npm pack` refuses modules that are
  missing, built without the patched WebGPU kernel, without the runtime teardown or with unfixed glue (`prepack`).
- `CHANGELOG.md` has the heading `## <version> (<date>)` with the date of the release.
- The version is the same in `package.json`, `package-lock.json`, `src/index.js` (`VERSION`), the first line of
  `types/index.d.ts` and the README's first paragraph.

```bash
npm test                          # unit tests, the public API listing included
npm run test:bundlers
npm run test:integration          # needs CZECHSPEECH_MODEL, CZECHSPEECH_TEST_AUDIO, CZECHSPEECH_TEST_MANIFEST (README)
npm run test:e2e                  # headless Chromium, CPU and WebGPU; same variables
npm run check:tarball             # packs, installs the tarball into an empty project, imports it, type-checks a consumer
git status --short                # must print nothing
```

`npm run check:tarball` leaves `moravianai-czechspeech-<version>.tgz` in the repository (ignored by git) and lists
its files: no test, script or native source may be among them.

## 2. Make the repository public and push

`package.json` (`repository`) and the README link to <https://github.com/moravian-ai/czechspeech-web>. Make the
repository public in GitHub's settings (Settings, General, Danger Zone, Change visibility) after a last look at
the history, the branches on the remote (`main`, `webgpu`) and the issues, then:

```bash
git push origin main
```

## 3. npm

```bash
npm publish --dry-run             # the same file list as check:tarball, and "public" access
npm publish                       # asks for the one-time password; publishConfig.access is "public"
```

`npm publish` runs `prepack` again and uploads exactly the files listed in `package.json` (`files`). There is no
provenance attestation: that needs a publish from GitHub Actions (trusted publishing), which can be set up for the
next version once the repository is public.

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
git tag -a v<version> -m "@moravianai/czechspeech <version>"
git push origin v<version>
```

## What the tools do, and what they leave to you

| step | who |
|---|---|
| `prepack`: the worklet is up to date, every listed file exists, the VAD hash, the versions agree, no build-machine path, the modules are the patched and fixed ones | npm, on `pack` and `publish` |
| `check:tarball`: installs the tarball into an empty project, imports it, type-checks a consumer | you run it (step 1) |
| building the WebAssembly modules; the browser and integration tests with the model | you (step 1) |
| making the repository public, `git push`, `npm publish` with the one-time password, the tag | you |
