import ReactMarkdown from "react-markdown";

interface Props {
  content: string;
  isComplete: boolean;
}

export function StreamingMessage({ content, isComplete }: Props) {
  return (
    <div className="flex justify-start">
      <div className="max-w-[80%] px-3 py-2 rounded-lg text-sm bg-gray-100 text-gray-800">
        <div className="markdown-content">
          <ReactMarkdown>{content}</ReactMarkdown>
          {!isComplete && (
            <span className="inline-block w-2 h-4 bg-gray-400 animate-pulse ml-0.5" />
          )}
        </div>
      </div>
    </div>
  );
}
