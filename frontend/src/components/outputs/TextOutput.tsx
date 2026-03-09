import ReactMarkdown from "react-markdown";

interface Props {
  content: string;
}

export function TextOutput({ content }: Props) {
  return (
    <div className="markdown-content text-sm py-2">
      <ReactMarkdown>{content}</ReactMarkdown>
    </div>
  );
}
