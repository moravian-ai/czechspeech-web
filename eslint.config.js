// ESLint flat config: the recommended rules for the SDK's JavaScript sources, tests and scripts; no warnings allowed.
import js from '@eslint/js';
import globals from 'globals';

export default [
  { ignores: ['node_modules/**', 'native/**', 'src/wasm/**', 'src/speakers/ort/**', 'src/assets/**', 'examples/**/vendor/**', 'test/speakers/page/**', 'test/speakers/ref/**'] },
  js.configs.recommended,
  {
    languageOptions: { ecmaVersion: 2024, sourceType: 'module', globals: { ...globals.node, ...globals.browser, ...globals.worker } },
  },
  { files: ['**/*.js', '**/*.mjs'], rules: {} },
];
