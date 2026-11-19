// The CHANGELOG heading of a version is "## <version> (YYYY-MM-DD)" with a real date once it is released; the
// placeholder "(YYYY-MM-DD)" (or a date that does not exist) must not reach a package.
// Returns null when the heading is fine, else a message.
export function changelogDateProblem(text, version) {
  const start = `## ${version} (`;
  const heading = text.split('\n').find((line) => line.startsWith(start));
  if (heading === undefined) return `CHANGELOG.md has no heading "## ${version} (date)"`;
  const inner = heading.slice(start.length).split(')')[0];
  const d = /^(\d{4})-(\d{2})-(\d{2})$/.exec(inner);
  const date = d ? new Date(Date.UTC(+d[1], +d[2] - 1, +d[3])) : null;
  if (!date || date.getUTCFullYear() !== +d[1] || date.getUTCMonth() !== +d[2] - 1 || date.getUTCDate() !== +d[3]) {
    return `CHANGELOG.md heading of ${version} has no real date ("${inner}"): fill it in on the day of the release`;
  }
  return null;
}
