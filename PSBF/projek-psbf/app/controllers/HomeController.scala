package controllers

import javax.inject._
import play.api.mvc._

@Singleton
class HomeController @Inject()(val controllerComponents: ControllerComponents) extends BaseController {

  // Halaman utama
  def index() = Action { implicit request: Request[AnyContent] =>
    Ok(views.html.index())
  }

  // Route baru untuk /main
  def main() = Action { implicit request: Request[AnyContent] =>
    Ok(views.html.main("Welcome to Play Framework"))
  }
}
