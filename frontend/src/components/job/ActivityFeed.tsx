import { useRef, useEffect } from 'react';
import { AgentEvent } from '../../services/api';
import { MAX_AGENT_EVENTS } from '../../config/constants';

interface ActivityFeedProps {
  events: AgentEvent[];
}

export default function ActivityFeed({ events }: ActivityFeedProps) {
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [events.length]);

  if (events.length === 0) {
    return (
      <div className="border-t border-ink-100 pt-4 mt-4">
        <p className="font-mono text-xs text-ink-300">Waiting for agent activity...</p>
      </div>
    );
  }

  return (
    <div className="border-t border-ink-100 pt-4 mt-4">
      <h3 className="text-xs font-medium text-ink-500 uppercase tracking-wider mb-3">Activity Log</h3>
      <div ref={scrollRef} className="max-h-48 overflow-y-auto space-y-1">
        {events.slice(-MAX_AGENT_EVENTS).map((event, i) => {
          const time = new Date(event.timestamp).toLocaleTimeString('en-US', {
            hour12: false, hour: '2-digit', minute: '2-digit', second: '2-digit'
          });
          const action = event.event_type === 'agent_started' ? 'started' :
                        event.event_type === 'agent_completed' ? 'completed' : event.event_type;

          return (
            <div key={i} className="flex items-baseline gap-3 animate-slide-in-left font-mono text-xs">
              <span className="text-ink-300 flex-shrink-0">{time}</span>
              <span className="text-accent flex-shrink-0">{event.agent_name}</span>
              <span className="text-ink-500">{action}</span>
            </div>
          );
        })}
      </div>
    </div>
  );
}
