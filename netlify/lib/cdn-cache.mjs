// Caching riddle-lab API responses on Netlify's CDN, so repeat views don't wake the database.
//
// A cacheable response is kept by the CDN (not by browsers) until it is purged by tag; the
// one-day limit is only a safety net in case a purge is missed. `durable` shares the cached
// copy across edge locations, so each change costs about one function call and one query.
// Every new deploy starts with an empty cache.
import { purgeCache } from '@netlify/cache';

export const LEADERBOARD_TAG = 'riddle-leaderboard';
export const jobTag = (id) => `riddle-job-${id}`;

export const cachedHeaders = (tag) => ({
  // Browsers always check back; the CDN answers them from its copy.
  'Cache-Control': 'public, max-age=0, must-revalidate',
  'Netlify-CDN-Cache-Control': 'public, durable, s-maxage=86400',
  'Netlify-Cache-Tag': tag,
});

// Purging takes a few seconds to reach every edge. A failed purge is logged, not thrown: the
// data is already saved, and the cached copy expires within a day anyway.
export async function purge(...tags) {
  try {
    await purgeCache({ tags });
  } catch (e) {
    console.error(`Cache purge failed for ${tags.join(', ')}: ${e?.message || e}`);
  }
}
