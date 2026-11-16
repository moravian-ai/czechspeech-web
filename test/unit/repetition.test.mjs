// The repetition-loop guard on recognition results: the same cases as the Android SDK's RepetitionGuardTest.kt.
// The loop is a real one: stop() 3.5 s into ParCzech4Speech segment test_0000-0000000333 (CC BY 4.0), cut mid-word.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { describeLoops, removeRepetitionLoops } from '../../src/repetition.js';

const PHRASE = 'se státní podporou v roce 2012 až';
const LOOP = 'pojišťování vývozu se státní podporou v roce 2012 až se státní podporou v roce 2012 až se státní podporou ' +
  'v roce 2012 až se státní podporou v roce 2012 až se st';
const CLEANED = 'pojišťování vývozu se státní podporou v roce 2012 až';

const unchanged = (text) => {
  const r = removeRepetitionLoops(text);
  assert.equal(r.text, text);
  assert.deepEqual(r.loops, []);
};

test('the loop after stop() mid-word keeps the first copy and drops the repeats and the cut-off copy', () => {
  const r = removeRepetitionLoops(LOOP);
  assert.equal(r.text, CLEANED);
  assert.deepEqual(r.loops, [{ words: 7, repeats: 3, partial: true }]);
  assert.equal(describeLoops(r.loops), 'repetition loop removed (7 words, 3 repeats, partial copy at the end)');
});

test('the same loop run on to the token limit', () => {
  const long = `pojišťování vývozu ${Array(9).fill(PHRASE).join(' ')} se státní podp`;
  const r = removeRepetitionLoops(long);
  assert.equal(r.text, CLEANED);
  assert.deepEqual(r.loops, [{ words: 7, repeats: 8, partial: true }]);
});

test('normal text stays untouched', () => {
  unchanged('pojišťování vývozu se státní podporou v roce 2012, sněmovní tisk 91. Děkuji.');
  unchanged('ne, ne, ne');
  unchanged('velmi děkuji, velmi děkuji');
  unchanged('1, 2, 3, 4, 5');
  unchanged('to je to');
  unchanged('Pane předsedající, já jsem říkal, já jsem říkal, že ten zákon nebude fungovat.');
});

test('a single repeated word stays, however often it repeats', () => {
  unchanged('Ne, ne, ne, ne, ne, ne, ne, ne.');
  unchanged('ne ne ne ne ne ne ne ne ne ne ne ne ne');
});

test('a loop in the middle is collapsed and the text after it is kept', () => {
  const r = removeRepetitionLoops('Návrh zákona o podpoře bydlení o podpoře bydlení o podpoře bydlení projednal výbor pro sociální politiku.');
  assert.equal(r.text, 'Návrh zákona o podpoře bydlení projednal výbor pro sociální politiku.');
  assert.deepEqual(r.loops, [{ words: 3, repeats: 2, partial: false }]);
  assert.equal(describeLoops(r.loops), 'repetition loop removed (3 words, 2 repeats)');
});

test('words are compared without case and punctuation; the first copy keeps its own', () => {
  const r = removeRepetitionLoops('Děkuji vám. Děkuji vám, děkuji vám! DĚKUJI VÁM');
  assert.equal(r.text, 'Děkuji vám.');
  assert.deepEqual(r.loops, [{ words: 2, repeats: 3, partial: false }]);
});

test('the smallest phrase that explains a run wins', () => {
  const r = removeRepetitionLoops('ano ne ano ne ano ne ano ne ano ne ano ne');
  assert.equal(r.text, 'ano ne');
  assert.deepEqual(r.loops, [{ words: 2, repeats: 5, partial: false }]);
});

test('phrases of 2 to 12 words count, longer ones do not', () => {
  const twelve = 'jedna dvě tři čtyři pět šest sedm osm devět deset jedenáct dvanáct';
  const r = removeRepetitionLoops(`${twelve} ${twelve} ${twelve}`);
  assert.equal(r.text, twelve);
  assert.deepEqual(r.loops, [{ words: 12, repeats: 2, partial: false }]);
  const thirteen = `${twelve} třináct`;
  unchanged(`${thirteen} ${thirteen} ${thirteen}`);
  const two = removeRepetitionLoops('to je to je to je');
  assert.equal(two.text, 'to je');
});

test('a trailing partial copy is dropped only when it is a prefix of the phrase', () => {
  const cut = removeRepetitionLoops('děkuji vám děkuji vám děkuji vám děk');
  assert.equal(cut.text, 'děkuji vám');
  assert.deepEqual(cut.loops, [{ words: 2, repeats: 2, partial: true }]);
  const other = removeRepetitionLoops('děkuji vám děkuji vám děkuji vám dobrou noc');
  assert.equal(other.text, 'děkuji vám dobrou noc');
  assert.deepEqual(other.loops, [{ words: 2, repeats: 2, partial: false }]);
});

test('a loop of loops is removed too (the rule is applied until nothing changes)', () => {
  const inner = 'já jsem já jsem já jsem řekl';
  const r = removeRepetitionLoops(`${inner} ${inner} ${inner}`);
  assert.equal(r.text, 'já jsem řekl');
  assert.deepEqual(r.loops.map((l) => l.words), [2, 2, 2, 3]);
  assert.equal(describeLoops(r.loops),
    '4 repetition loops removed (2 words, 2 repeats; 2 words, 2 repeats; 2 words, 2 repeats; 3 words, 2 repeats)');
});

test('punctuation standing alone belongs to the word before it', () => {
  const r = removeRepetitionLoops('a tak dále – a tak dále – a tak dále – a pak konec');
  assert.equal(r.text, 'a tak dále – a pak konec');
  assert.deepEqual(r.loops, [{ words: 3, repeats: 2, partial: false }]);
});
