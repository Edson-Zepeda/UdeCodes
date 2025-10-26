import { readFileSync, writeFileSync, existsSync } from "node:fs";
import { resolve } from "node:path";

const FILES = [
  "node_modules/chart.js/dist/chunks/helpers.dataset.js",
  "node_modules/chart.js/dist/chunks/helpers.dataset.js.map"
];

const projectRoot = new URL("..", import.meta.url).pathname;

function stripSourceMapReference(filePath) {
  if (!existsSync(filePath)) {
    return;
  }
  const contents = readFileSync(filePath, "utf8");
  const updated = contents.replace(/\/\/# sourceMappingURL=.*\n?$/gm, "");
  if (updated !== contents) {
    writeFileSync(filePath, updated, "utf8");
    console.log(`[fix-chartjs-sourcemap] Removed sourceMappingURL from ${filePath}`);
  }
}

try {
  const [helpersPath, mapPath] = FILES.map((relative) => resolve(projectRoot, relative));
  stripSourceMapReference(helpersPath);
  if (existsSync(mapPath)) {
    writeFileSync(mapPath, "", "utf8");
    console.log(`[fix-chartjs-sourcemap] Cleared problematic map ${mapPath}`);
  }
} catch (error) {
  console.warn("[fix-chartjs-sourcemap] Warning:", error);
}
