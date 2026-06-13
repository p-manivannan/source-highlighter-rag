import { useEffect, useMemo } from "react";
import { ComponentProps, Streamlit } from "streamlit-component-lib";

type Claim = {
  text: string;
  citation_ids: string[];
};

type Citation = {
  citation_id: string;
  citation_number: number;
  label: string;
};

type CitationAnswerArgs = {
  claims: Claim[];
  citations: Citation[];
};

function eventId(): string {
  if (typeof crypto.randomUUID === "function") {
    return crypto.randomUUID();
  }
  return `${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

export default function CitationAnswer({ args }: ComponentProps) {
  const { claims, citations } = args as CitationAnswerArgs;
  const citationLookup = useMemo(
    () => new Map(citations.map((citation) => [citation.citation_id, citation])),
    [citations],
  );

  useEffect(() => {
    Streamlit.setFrameHeight();
  });

  return (
    <div className="citation-answer">
      {claims.map((claim, claimIndex) => (
        <p className="answer-claim" key={`${claimIndex}-${claim.text}`}>
          <span>{claim.text}</span>
          {claim.citation_ids.map((citationId) => {
            const citation = citationLookup.get(citationId);
            if (!citation) {
              return null;
            }
            return (
              <button
                className="inline-citation"
                key={citationId}
                type="button"
                title={citation.label}
                aria-label={`Open source ${citation.citation_number}: ${citation.label}`}
                onClick={() =>
                  Streamlit.setComponentValue({
                    citation_id: citationId,
                    event_id: eventId(),
                  })
                }
              >
                [{citation.citation_number}]
              </button>
            );
          })}
        </p>
      ))}
    </div>
  );
}
