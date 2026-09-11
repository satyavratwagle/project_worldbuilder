import React from 'react';

export default function SentenceLink({ sentence, targetWords, onWordClick }) {
  if (!sentence) return null;

  const escapedWords = targetWords.map(w => w.replace(/[-\/\\^$*+?.()|[\]{}]/g, '\\$&'));
  const regex = new RegExp(`(${escapedWords.join('|')})`, 'g');
  const parts = sentence.split(regex);

  return (
    <p>
      {parts.map((part, index) => {
        if (targetWords.includes(part)) {
          return (
            <span
              key={index}
              onClick={() => onWordClick(part)}
              className="cursor-pointer text-blue-600 underline hover:text-blue-800 font-medium"
            >
              {part}
            </span>
          );
        }
        return part;
      })}
    </p>
  );
}