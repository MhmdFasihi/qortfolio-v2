import reflex as rx

# Explicitly disable the default sitemap plugin to silence warnings.
config = rx.Config(
    app_name="qortfolio_v2",
    disable_plugins=["reflex.plugins.sitemap.SitemapPlugin"],
)
