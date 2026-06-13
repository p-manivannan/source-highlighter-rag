import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  ComponentProps,
  Streamlit,
} from "streamlit-component-lib";
import { Document, Page, pdfjs } from "react-pdf";
import type { PDFDocumentProxy, PDFPageProxy } from "pdfjs-dist";

import "react-pdf/dist/Page/AnnotationLayer.css";
import "react-pdf/dist/Page/TextLayer.css";

pdfjs.GlobalWorkerOptions.workerSrc = new URL(
  "pdfjs-dist/build/pdf.worker.min.mjs",
  import.meta.url,
).toString();

type CharacterRef = {
  itemIndex: number;
  charIndex: number;
};

type ItemRange = {
  start: number;
  end: number;
};

type NormalizedText = {
  text: string;
  refs: Array<CharacterRef | null>;
};

type ViewerArgs = {
  pdf_base64: string;
  document_name: string;
  page_number: number;
  chunk_text: string;
  selection_key: string;
};

const MIN_SCALE = 0.7;
const MAX_SCALE = 2;
const SCALE_STEP = 0.15;

function canonicalCharacter(character: string): string {
  const punctuation: Record<string, string> = {
    "\u2018": "'",
    "\u2019": "'",
    "\u201c": '"',
    "\u201d": '"',
    "\u2013": "-",
    "\u2014": "-",
    "\u00a0": " ",
  };
  return punctuation[character] ?? character;
}

function normalizeWithMap(
  raw: string,
  rawRefs?: Array<CharacterRef | null>,
): NormalizedText {
  const text: string[] = [];
  const refs: Array<CharacterRef | null> = [];
  let index = 0;

  while (index < raw.length) {
    const character = canonicalCharacter(raw[index]).toLocaleLowerCase();
    const nextNonSpace = raw.slice(index + 1).search(/\S/);
    const nextCharacter =
      nextNonSpace >= 0
        ? canonicalCharacter(raw[index + 1 + nextNonSpace]).toLocaleLowerCase()
        : "";

    // PDF extraction commonly represents a line-wrapped word as "govern-\nment".
    if (
      character === "-" &&
      /^\s+$/.test(raw.slice(index + 1, index + 1 + Math.max(nextNonSpace, 0))) &&
      /^[a-z]$/.test(nextCharacter)
    ) {
      index += 1 + Math.max(nextNonSpace, 0);
      continue;
    }

    if (/\s/.test(character)) {
      if (text.length > 0 && text[text.length - 1] !== " ") {
        text.push(" ");
        refs.push(rawRefs?.[index] ?? null);
      }
      index += 1;
      while (index < raw.length && /\s/.test(raw[index])) {
        index += 1;
      }
      continue;
    }

    const decomposed = character.normalize("NFKD").replace(/\p{M}/gu, "");
    for (const outputCharacter of decomposed) {
      text.push(outputCharacter);
      refs.push(rawRefs?.[index] ?? null);
    }
    index += 1;
  }

  if (text[text.length - 1] === " ") {
    text.pop();
    refs.pop();
  }

  return { text: text.join(""), refs };
}

function buildPageText(items: Array<{ str: string; hasEOL?: boolean }>): NormalizedText {
  const rawCharacters: string[] = [];
  const rawRefs: Array<CharacterRef | null> = [];

  items.forEach((item, itemIndex) => {
    Array.from(item.str).forEach((character, charIndex) => {
      rawCharacters.push(character);
      rawRefs.push({ itemIndex, charIndex });
    });
    // PDF.js already emits explicit whitespace items. Adding a space after every
    // item corrupts words that a PDF font encodes as adjacent fragments, such as
    // "a" + "nd" or "t" + "he".
    if (item.hasEOL) {
      rawCharacters.push("\n");
      rawRefs.push(null);
    }
  });

  return normalizeWithMap(rawCharacters.join(""), rawRefs);
}

function locateHighlight(
  pageText: NormalizedText,
  chunkText: string,
): Map<number, ItemRange> {
  const normalizedChunk = normalizeWithMap(chunkText).text.trim();
  if (!normalizedChunk) {
    return new Map();
  }

  let matchedText = normalizedChunk;
  let matchStart = pageText.text.indexOf(matchedText);

  // Long chunks can end with text from an adjacent extraction boundary. Keep
  // shrinking at word boundaries so the cited passage still receives a useful,
  // substantial highlight instead of failing the entire match.
  while (matchStart < 0 && matchedText.length >= 120) {
    const previousBoundary = matchedText.lastIndexOf(" ");
    if (previousBoundary < 0) {
      break;
    }
    matchedText = matchedText.slice(0, previousBoundary).trimEnd();
    matchStart = pageText.text.indexOf(matchedText);
  }

  if (matchStart < 0) {
    return new Map();
  }

  const ranges = new Map<number, ItemRange>();
  const matchRefs = pageText.refs.slice(
    matchStart,
    matchStart + matchedText.length,
  );

  matchRefs.forEach((ref) => {
    if (!ref) {
      return;
    }
    const current = ranges.get(ref.itemIndex);
    if (!current) {
      ranges.set(ref.itemIndex, {
        start: ref.charIndex,
        end: ref.charIndex + 1,
      });
      return;
    }
    current.start = Math.min(current.start, ref.charIndex);
    current.end = Math.max(current.end, ref.charIndex + 1);
  });

  return ranges;
}

function escapeHtml(value: string): string {
  return value
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}

function SourceViewer({ args, width }: ComponentProps) {
  const {
    pdf_base64: pdfBase64,
    document_name: documentName,
    page_number: requestedPage,
    chunk_text: chunkText,
    selection_key: selectionKey,
  } = args as ViewerArgs;

  const [document, setDocument] = useState<PDFDocumentProxy | null>(null);
  const [pageNumber, setPageNumber] = useState(Math.max(1, requestedPage));
  const [scale, setScale] = useState(1);
  const [highlightRanges, setHighlightRanges] = useState<Map<number, ItemRange>>(
    new Map(),
  );
  const [analysisPending, setAnalysisPending] = useState(true);
  const [highlightFound, setHighlightFound] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const viewerRef = useRef<HTMLDivElement>(null);

  const pdfData = useMemo(() => {
    const binary = atob(pdfBase64);
    const bytes = new Uint8Array(binary.length);
    for (let index = 0; index < binary.length; index += 1) {
      bytes[index] = binary.charCodeAt(index);
    }
    return { data: bytes };
  }, [pdfBase64]);

  useEffect(() => {
    setDocument(null);
    setPageNumber(Math.max(1, requestedPage));
    setScale(1);
    setHighlightRanges(new Map());
    setAnalysisPending(true);
    setHighlightFound(false);
    setError(null);
  }, [selectionKey, requestedPage]);

  useEffect(() => {
    Streamlit.setFrameHeight();
  });

  useEffect(() => {
    if (!document) {
      return;
    }

    const boundedPage = Math.min(Math.max(1, pageNumber), document.numPages);
    if (boundedPage !== pageNumber) {
      setPageNumber(boundedPage);
      return;
    }

    let cancelled = false;
    setAnalysisPending(true);
    setError(null);

    document
      .getPage(boundedPage)
      .then((page: PDFPageProxy) => page.getTextContent())
      .then((content) => {
        if (cancelled) {
          return;
        }
        const items = content.items
          .filter(
            (item): item is Extract<typeof item, { str: string }> => "str" in item,
          )
          .map((item) => ({ str: item.str, hasEOL: item.hasEOL }));
        const ranges = locateHighlight(buildPageText(items), chunkText);
        setHighlightRanges(ranges);
        setHighlightFound(ranges.size > 0);
        setAnalysisPending(false);
      })
      .catch(() => {
        if (!cancelled) {
          setError("The PDF page text could not be analyzed.");
          setAnalysisPending(false);
        }
      });

    return () => {
      cancelled = true;
    };
  }, [document, pageNumber, chunkText]);

  const renderText = useCallback(
    ({ str, itemIndex }: { str: string; itemIndex: number }) => {
      const range = highlightRanges.get(itemIndex);
      if (!range) {
        return escapeHtml(str);
      }
      return [
        escapeHtml(str.slice(0, range.start)),
        '<mark class="source-highlight" data-source-highlight="true">',
        escapeHtml(str.slice(range.start, range.end)),
        "</mark>",
        escapeHtml(str.slice(range.end)),
      ].join("");
    },
    [highlightRanges],
  );

  const handleTextLayerReady = useCallback(() => {
    requestAnimationFrame(() => {
      const firstHighlight = viewerRef.current?.querySelector(
        "[data-source-highlight='true']",
      );
      firstHighlight?.scrollIntoView({ behavior: "smooth", block: "center" });
      Streamlit.setFrameHeight();
    });
  }, []);

  const handleDocumentLoad = useCallback(
    (loadedDocument: PDFDocumentProxy) => {
      setDocument(loadedDocument);
      setPageNumber(Math.min(Math.max(1, requestedPage), loadedDocument.numPages));
    },
    [requestedPage],
  );

  const pageWidth = Math.max(280, Math.min((width || 500) - 28, 760));

  return (
    <section className="source-viewer" ref={viewerRef}>
      <header className="viewer-header">
        <div className="document-details">
          <strong title={documentName}>{documentName}</strong>
          <span>
            Page {pageNumber}
            {document ? ` of ${document.numPages}` : ""}
          </span>
        </div>
        <div className="viewer-controls" aria-label="PDF controls">
          <button
            type="button"
            onClick={() => setPageNumber((page) => Math.max(1, page - 1))}
            disabled={!document || pageNumber <= 1}
            aria-label="Previous page"
          >
            Prev
          </button>
          <button
            type="button"
            onClick={() =>
              setPageNumber((page) =>
                document ? Math.min(document.numPages, page + 1) : page,
              )
            }
            disabled={!document || pageNumber >= document.numPages}
            aria-label="Next page"
          >
            Next
          </button>
          <button
            type="button"
            onClick={() =>
              setScale((current) => Math.max(MIN_SCALE, current - SCALE_STEP))
            }
            disabled={scale <= MIN_SCALE}
            aria-label="Zoom out"
          >
            -
          </button>
          <span className="zoom-value">{Math.round(scale * 100)}%</span>
          <button
            type="button"
            onClick={() =>
              setScale((current) => Math.min(MAX_SCALE, current + SCALE_STEP))
            }
            disabled={scale >= MAX_SCALE}
            aria-label="Zoom in"
          >
            +
          </button>
        </div>
      </header>

      {error && <div className="viewer-notice error">{error}</div>}
      {!analysisPending && !highlightFound && !error && (
        <div className="viewer-notice warning">
          The cited page opened, but the source text location could not be matched.
        </div>
      )}

      <div className="pdf-scroll-area">
        <Document
          file={pdfData}
          onLoadSuccess={handleDocumentLoad}
          onLoadError={() => setError("The cited PDF could not be loaded.")}
          loading={<div className="loading-state">Loading PDF...</div>}
        >
          {!analysisPending && (
            <Page
              pageNumber={pageNumber}
              width={pageWidth}
              scale={scale}
              renderTextLayer
              renderAnnotationLayer
              customTextRenderer={renderText}
              onRenderTextLayerSuccess={handleTextLayerReady}
              loading={<div className="loading-state">Rendering page...</div>}
            />
          )}
        </Document>
        {analysisPending && document && (
          <div className="loading-state">Locating cited text...</div>
        )}
      </div>
    </section>
  );
}

export default SourceViewer;
