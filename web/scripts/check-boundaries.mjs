import ts from "typescript";
import { readdirSync, readFileSync } from "node:fs";
import { dirname, relative, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const root = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const configFile = ts.readConfigFile(resolve(root, "tsconfig.app.json"), ts.sys.readFile);
const config = ts.parseJsonConfigFileContent(configFile.config, ts.sys, root);
const files = (dir) => readdirSync(dir, { withFileTypes: true }).flatMap((entry) =>
  entry.isDirectory() ? files(resolve(dir, entry.name)) : /\.tsx?$/.test(entry.name) ? [resolve(dir, entry.name)] : []);
const graph = new Map();
const errors = [];
const feature = (path) => /\/features\/(chat|analysis)\//.exec(path)?.[1];
for (const path of files(resolve(root, "src"))) {
  const source = ts.createSourceFile(path, readFileSync(path, "utf8"), ts.ScriptTarget.Latest, true);
  const imports = [];
  const visit = (node) => {
    let specifier;
    if (ts.isImportDeclaration(node) || ts.isExportDeclaration(node)) specifier = node.moduleSpecifier;
    if (ts.isImportTypeNode(node) && ts.isLiteralTypeNode(node.argument)) specifier = node.argument.literal;
    if (ts.isCallExpression(node) && (node.expression.kind === ts.SyntaxKind.ImportKeyword || node.expression.getText(source) === "require")) {
      specifier = node.arguments[0];
      if (feature(path) && (!specifier || !ts.isStringLiteralLike(specifier))) errors.push(`${relative(root, path)}: nonliteral dynamic import cannot be checked`);
    }
    if (specifier && ts.isStringLiteralLike(specifier)) {
      const target = ts.resolveModuleName(specifier.text, path, config.options, ts.sys).resolvedModule?.resolvedFileName;
      if (target?.startsWith(resolve(root, "src"))) imports.push(target);
    }
    ts.forEachChild(node, visit);
  };
  visit(source);
  graph.set(path, imports);
}
for (const path of graph.keys()) {
  const owner = feature(path);
  if (!owner) continue;
  const seen = new Set();
  const pending = [[path, [path]]];
  while (pending.length) {
    const [current, chain] = pending.pop();
    if (seen.has(current)) continue;
    seen.add(current);
    if (feature(current) && feature(current) !== owner) {
      errors.push(`Forbidden dependency: ${chain.map((p) => relative(root, p)).join(" -> ")}`);
      break;
    }
    for (const target of graph.get(current) ?? []) pending.push([target, [...chain, target]]);
  }
}
if (errors.length) {
  console.error(errors.join("\n"));
  process.exitCode = 1;
} else console.log(`TypeScript component boundaries pass (${graph.size} modules).`);
