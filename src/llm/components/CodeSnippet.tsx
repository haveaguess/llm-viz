import React, { useState } from 'react';

// Simple Python syntax highlighter for nanoGPT code snippets
function highlightPython(code: string): React.ReactNode[] {
    let lines = code.split('\n');
    return lines.map((line, i) => {
        let parts: React.ReactNode[] = [];
        let remaining = line;
        let keyId = 0;

        // Process the line character by character with regex matches
        while (remaining.length > 0) {
            let match: RegExpMatchArray | null = null;
            let matchType = '';

            // Comment
            if ((match = remaining.match(/^(#.*)/))) {
                matchType = 'comment';
            }
            // String (single or double quoted)
            else if ((match = remaining.match(/^((?:f?(?:"(?:[^"\\]|\\.)*"|'(?:[^'\\]|\\.)*')))/))) {
                matchType = 'string';
            }
            // Decorator
            else if ((match = remaining.match(/^(@\w+)/))) {
                matchType = 'decorator';
            }
            // Keyword
            else if ((match = remaining.match(/^(def|class|return|if|else|elif|for|in|import|from|as|assert|not|and|or|None|True|False|self|with|lambda|yield|raise|try|except|finally|while|break|continue|pass|is|del|global|nonlocal|async|await)\b/))) {
                matchType = 'keyword';
            }
            // Built-in / type
            else if ((match = remaining.match(/^(nn|torch|F|math|int|float|bool|str|list|dict|tuple|set|print|len|range|sum|super|isinstance|hasattr|getattr)\b/))) {
                matchType = 'builtin';
            }
            // Number
            else if ((match = remaining.match(/^(\d+\.?\d*(?:e[+-]?\d+)?)/))) {
                matchType = 'number';
            }
            // Function call
            else if ((match = remaining.match(/^(\w+)(?=\()/))) {
                matchType = 'function';
            }
            // Regular text (consume one character or a word)
            else if ((match = remaining.match(/^(\w+|[^\w])/))) {
                matchType = 'text';
            }

            if (match) {
                let text = match[0];
                let className = matchType !== 'text' ? `code-${matchType}` : undefined;
                if (className) {
                    parts.push(<span key={keyId++} className={className}>{text}</span>);
                } else {
                    parts.push(<span key={keyId++}>{text}</span>);
                }
                remaining = remaining.slice(text.length);
            } else {
                parts.push(remaining);
                break;
            }
        }

        return <React.Fragment key={i}>
            {i > 0 && '\n'}
            {parts}
        </React.Fragment>;
    });
}

export interface CodeSnippetProps {
    code: string;
    filename?: string;
    lineStart?: number;
    collapsed?: boolean;
}

export const CodeSnippet: React.FC<CodeSnippetProps> = ({ code, filename, lineStart, collapsed: initialCollapsed }) => {
    let [isCollapsed, setIsCollapsed] = useState(initialCollapsed ?? false);

    let trimmedCode = code.replace(/^\n+/, '').replace(/\n+$/, '');

    return <div className="code-snippet">
        {filename && <div className="code-snippet-header" onClick={() => setIsCollapsed(!isCollapsed)}>
            <span className="code-snippet-toggle">{isCollapsed ? '▶' : '▼'}</span>
            <span className="code-snippet-filename">{filename}</span>
            {lineStart && <span className="code-snippet-line">:{lineStart}</span>}
        </div>}
        {!isCollapsed && <pre className="code-snippet-code">
            <code>{highlightPython(trimmedCode)}</code>
        </pre>}
    </div>;
};

// Helper to create an embeddable code snippet for use in commentary
export function codeSnippet(code: string, filename?: string, lineStart?: number, collapsed?: boolean) {
    return { insertInline: <CodeSnippet code={code} filename={filename} lineStart={lineStart} collapsed={collapsed} /> };
}
