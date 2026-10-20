import { siteLinks, type SiteLink } from "../predictor-ui";

// Every Predictor site, for the family switcher: labels and links come from the shared list in predictor-ui, so a
// relabel is one change in the hub package. Change the host here when the real domain lands.
const HOST = "40-160-91-131.sslip.io";
export const SITES: SiteLink[] = siteLinks("pl", HOST);
