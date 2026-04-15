import { useState } from "react";
import { Code2, ChevronDown, ChevronRight, Copy, Check } from "lucide-react";

export default function CodeBlock({ sql }) {
  const [open, setOpen] = useState(false);
  const [copied, setCopied] = useState(false);

  const handleCopy = (e) => {
    e.stopPropagation();
    navigator.clipboard.writeText(sql);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="mt-3 border border-gray-800 rounded-lg overflow-hidden">
      <button
        onClick={() => setOpen(!open)}
        className="w-full flex items-center gap-2 px-3 py-2 text-xs text-gray-400 hover:text-gray-200 hover:bg-gray-800/40 transition-colors"
      >
        {open ? <ChevronDown size={12} /> : <ChevronRight size={12} />}
        <Code2 size={12} />
        <span>SQL Query</span>
        <button
          onClick={handleCopy}
          className="ml-auto flex items-center gap-1 hover:text-indigo-400 transition-colors"
        >
          {copied ? <Check size={11} className="text-emerald-400" /> : <Copy size={11} />}
          <span>{copied ? "Copied" : "Copy"}</span>
        </button>
      </button>
      {open && (
        <pre className="px-4 py-3 bg-[#0c1222] text-[13px] text-indigo-300 font-mono overflow-x-auto leading-relaxed border-t border-gray-800">
          {sql}
        </pre>
      )}
    </div>
  );
}
