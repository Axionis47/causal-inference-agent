import { useState, ReactNode } from 'react';
import glossary from '../../utils/glossary';

interface TooltipProps {
  term: string;
  children: ReactNode;
}

export default function Tooltip({ term, children }: TooltipProps) {
  const [show, setShow] = useState(false);
  const definition = glossary[term];
  if (!definition) return <>{children}</>;

  return (
    <span
      className="relative inline-flex items-center cursor-help"
      style={{ borderBottom: '1px dotted #9ca3af' }}
      onMouseEnter={() => setShow(true)}
      onMouseLeave={() => setShow(false)}
    >
      {children}
      {show && (
        <span
          className="absolute z-50 text-xs text-white rounded-lg shadow-lg"
          style={{
            bottom: 'calc(100% + 8px)',
            left: '50%',
            transform: 'translateX(-50%)',
            backgroundColor: '#1f2937',
            padding: '8px 12px',
            width: '260px',
            lineHeight: '1.5',
            pointerEvents: 'none',
          }}
        >
          {definition}
          {/* Arrow */}
          <span
            style={{
              position: 'absolute',
              bottom: '-6px',
              left: '50%',
              transform: 'translateX(-50%)',
              width: 0,
              height: 0,
              borderLeft: '6px solid transparent',
              borderRight: '6px solid transparent',
              borderTop: '6px solid #1f2937',
            }}
          />
        </span>
      )}
    </span>
  );
}
