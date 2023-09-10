import shaderUtil from "./shaderUtil.wgsl";

// A super simple crappy preprocessor for includes
export function PreprocessWgsl(input: string, defines: Array<string>): string {
  const includes: { [key: string]: string } = {
    "shaderUtil.wgsl": shaderUtil,
  };

  var last_text = input;

  while (true) {
    const lines = last_text.split(/\r?\n/);

    const new_lines = [];
    const takeBranch = [];
    const hasTakenBranch = [];

    var define = "";

    for (var line of lines) {
      const trimLine = line.trimStart();

      if (trimLine.startsWith("//")) {
        // filter
      } else if (trimLine.startsWith("#if ")) {
        define = line.substring(4).trim();

        // strip off comments
        if (define.indexOf("//") !== -1) {
          define = define.substring(0, define.indexOf("//")).trim();
        }

        if (defines.includes(define)) {
          takeBranch.push(true);
          hasTakenBranch.push(true);
        } else {
          takeBranch.push(false);
          hasTakenBranch.push(false);
        }
      } else if (trimLine.startsWith("#elseif ")) {
        define = line.substring(7).trim();

        // strip off comments
        if (define.indexOf("//") !== -1) {
          define = define.substring(0, define.indexOf("//")).trim();
        }

        takeBranch.pop();

        if (hasTakenBranch[hasTakenBranch.length - 1]) {
          takeBranch.push(false);
        } else {
          if (defines.includes(define)) {
            takeBranch.push(true);
            hasTakenBranch[hasTakenBranch.length - 1] = true;
          } else {
            takeBranch.push(false);
          }
        }
      } else if (trimLine.startsWith("#else")) {
        define = "";

        takeBranch.pop();

        if (!hasTakenBranch[hasTakenBranch.length - 1]) {
          takeBranch.push(true);
          hasTakenBranch[hasTakenBranch.length - 1] = true;
        } else {
          takeBranch.push(false);
        }
      } else if (trimLine.startsWith("#endif")) {
        takeBranch.pop();
        hasTakenBranch.pop();
      } else {
        if (takeBranch.length === 0) {
          new_lines.push(line);
        } else {
          if (takeBranch[takeBranch.length - 1]) {
            new_lines.push(line);
          }
        }
      }
    }

    var new_text = "";

    for (var line of new_lines) {
      const trimLine = line.trimStart();

      if (trimLine.startsWith("#include ")) {
        // strip off #include
        var includeName = trimLine.substring(9).trim();

        // strip off brackets
        includeName = includeName.substring(1, includeName.length - 1);

        const includeText = includes[includeName];

        new_text += includeText;

        if (!includeText.endsWith("\n")) new_text += "\n";
      } else {
        new_text += line + "\n";
      }
    }

    new_text = new_text.trim();

    if (new_text === last_text) break;

    last_text = new_text;
  }

  return last_text;
}