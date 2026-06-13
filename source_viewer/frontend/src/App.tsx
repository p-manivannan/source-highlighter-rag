import { ComponentProps, withStreamlitConnection } from "streamlit-component-lib";

import CitationAnswer from "./CitationAnswer";
import SourceViewer from "./SourceViewer";

function App(props: ComponentProps) {
  if (props.args.view === "citation_answer") {
    return <CitationAnswer {...props} />;
  }
  return <SourceViewer {...props} />;
}

export default withStreamlitConnection(App);
