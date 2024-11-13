<?xml version="1.0" encoding="UTF-8"?>
<xsl:stylesheet xmlns:xsl="http://www.w3.org/1999/XSL/Transform" version="1.0">
<xsl:output method="html" indent="yes"/>
<xsl:template match="/">
<html>
    <body>
       <header><h1>Le seigneur des Anneaux</h1></header>
       <h2>
       <xsl:apply-templates select="//film" />
       </h2>
    </body>
</html>
</xsl:template>
<xsl:template match="//film" >
  <h2><xsl:value-of select="title" /></h2>
  <xsl:value-of select="director/attribute::label" />: <xsl:value-of select="director" /><br/>
  <xsl:value-of select="release-date/attribute::label" />: <xsl:value-of select="release-date" /><br/>
  <xsl:value-of select="box-office/attribute::label" />: <xsl:value-of select="box-office" /><br/>
  <ul>
  <li><xsl:value-of select="roles/role/attribute::character" /></li>
  </ul>
</xsl:template>
</xsl:stylesheet>