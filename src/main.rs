use bevy::{
    diagnostic::FrameTimeDiagnosticsPlugin,
    prelude::*,
    text::FontSmoothing,
    window::{PresentMode, WindowResolution, WindowTheme},
};
use fps_overlay::{FpsOverlayConfig, FpsOverlayPlugin};
use image::{ImageBuffer, Rgba};
use systems::*;
use xcap::Monitor;

fn main() -> AppExit {
    // TODO: this should always run in the background listening for input, and launch a new window when we make a new capture
    let images = match snapshot() {
        Ok(images) => images,
        Err(err) => {
            eprintln!("Encountered an error: {err}");
            return AppExit::error();
        }
    };
    if images.is_empty() {
        eprintln!("Encountered an error: No monitor image was captured");
        return AppExit::error();
    }
    let mut min_x = i32::MAX;
    let mut max_x = i32::MIN;
    let mut min_y = i32::MAX;
    let mut max_y = i32::MIN;
    for image in &images {
        min_x = min_x.min(image.x);
        max_x = max_x.max(image.x + image.image.width() as i32);
        min_y = min_y.min(image.y);
        max_y = max_y.max(image.y + image.image.height() as i32);
    }
    App::new()
        .add_plugins((
            DefaultPlugins.set(WindowPlugin {
                primary_window: Some(Window {
                    title: "Embershot".to_string(),
                    resolution: WindowResolution::new((max_x - min_x) as _, (max_y - min_y) as _),
                    present_mode: PresentMode::AutoVsync,
                    // Tells Wasm to resize the window according to the available canvas
                    fit_canvas_to_parent: true,
                    // Tells Wasm not to override default event handling, like F5, Ctrl+R etc.
                    prevent_default_event_handling: false,
                    window_theme: Some(WindowTheme::Dark),
                    // This will spawn an invisible window
                    // The window will be made visible in the make_visible() system after 3 frames.
                    // This is useful when you want to avoid the white window that shows up before the GPU is ready to render the app.
                    visible: false,
                    decorations: false,
                    position: WindowPosition::At((min_x, min_y).into()),
                    ..default()
                }),
                ..default()
            }),
            FrameTimeDiagnosticsPlugin::default(),
            FpsOverlayPlugin {
                config: FpsOverlayConfig {
                    text_config: TextFont {
                        // Here we define size of our overlay
                        font_size: 42.0,
                        // If we want, we can use a custom font
                        font: default(),
                        // We could also disable font smoothing,
                        font_smoothing: FontSmoothing::default(),
                        ..default()
                    },
                    // We can also change color of the overlay
                    text_color: Color::srgb(0.0, 1.0, 0.0),
                    // We can also set the refresh interval for the FPS counter
                    refresh_interval: core::time::Duration::from_millis(100),
                    enabled: true,
                },
            },
        ))
        .insert_resource(Once::new(images))
        .insert_resource(Extents {
            min_x,
            max_x,
            min_y,
            max_y,
        })
        .insert_resource(Scale(0.0))
        .init_resource::<WorldCursor>()
        .init_resource::<ViewportCursor>()
        .add_systems(
            Startup,
            (
                spawn_camera,
                spawn_monitor_images,
                setup_window,
                (spawn_target_ui, update_target_ui).chain(),
            ),
        )
        // TODO: debug possible one-frame delays on resize box
        .add_systems(
            Update,
            (
                make_window_visible,
                reset,
                toggle_pixelation,
                grab,
                zoom,
                update_global_cursors,
                // temp_create_on_click,
                (
                    update_target_viewport_rect,
                    resize_target,
                    update_target_viewport_rect,
                    update_target_ui,
                )
                    .chain()
                    .after(update_global_cursors)
                    .after(grab),
            ),
        )
        .run()
}

mod fps_overlay {
    //! Module containing logic for FPS overlay.

    use bevy::app::{Plugin, Startup, Update};
    use bevy::asset::Handle;
    use bevy::color::Color;
    use bevy::diagnostic::{DiagnosticsStore, FrameTimeDiagnosticsPlugin};
    use bevy::ecs::{
        change_detection::DetectChangesMut,
        component::Component,
        entity::Entity,
        prelude::Local,
        query::With,
        resource::Resource,
        schedule::{IntoScheduleConfigs, common_conditions::resource_changed},
        system::{Commands, Query, Res},
    };
    use bevy::render::view::Visibility;
    use bevy::text::{Font, TextColor, TextFont, TextSpan};
    use bevy::time::Time;
    use bevy::ui::{
        GlobalZIndex, Node, PositionType,
        widget::{Text, TextUiWriter},
    };
    use core::time::Duration;

    /// [`GlobalZIndex`] used to render the fps overlay.
    ///
    /// We use a number slightly under `i32::MAX` so you can render on top of it if you really need to.
    pub const FPS_OVERLAY_ZINDEX: i32 = i32::MAX - 32;

    /// A plugin that adds an FPS overlay to the Bevy application.
    ///
    /// This plugin will add the [`FrameTimeDiagnosticsPlugin`] if it wasn't added before.
    ///
    /// Note: It is recommended to use native overlay of rendering statistics when possible for lower overhead and more accurate results.
    /// The correct way to do this will vary by platform:
    /// - **Metal**: setting env variable `MTL_HUD_ENABLED=1`
    #[derive(Default)]
    pub struct FpsOverlayPlugin {
        /// Starting configuration of overlay, this can be later be changed through [`FpsOverlayConfig`] resource.
        pub config: FpsOverlayConfig,
    }

    impl Plugin for FpsOverlayPlugin {
        fn build(&self, app: &mut bevy::app::App) {
            // TODO: Use plugin dependencies, see https://github.com/bevyengine/bevy/issues/69
            if !app.is_plugin_added::<FrameTimeDiagnosticsPlugin>() {
                app.add_plugins(FrameTimeDiagnosticsPlugin::default());
            }
            app.insert_resource(self.config.clone())
                .add_systems(Startup, setup)
                .add_systems(
                    Update,
                    (
                        (customize_text, toggle_display)
                            .run_if(resource_changed::<FpsOverlayConfig>),
                        update_text,
                    ),
                );
        }
    }

    /// Configuration options for the FPS overlay.
    #[derive(Resource, Clone)]
    pub struct FpsOverlayConfig {
        /// Configuration of text in the overlay.
        pub text_config: TextFont,
        /// Color of text in the overlay.
        pub text_color: Color,
        /// Displays the FPS overlay if true.
        pub enabled: bool,
        /// The period after which the FPS overlay re-renders.
        ///
        /// Defaults to once every 100 ms.
        pub refresh_interval: Duration,
    }

    impl Default for FpsOverlayConfig {
        fn default() -> Self {
            FpsOverlayConfig {
                text_config: TextFont {
                    font: Handle::<Font>::default(),
                    font_size: 32.0,
                    ..Default::default()
                },
                text_color: Color::WHITE,
                enabled: true,
                refresh_interval: Duration::from_millis(100),
            }
        }
    }

    #[derive(Component)]
    struct FpsText;

    fn setup(mut commands: Commands, overlay_config: Res<FpsOverlayConfig>) {
        commands
            .spawn((
                Node {
                    // We need to make sure the overlay doesn't affect the position of other UI nodes
                    position_type: PositionType::Absolute,
                    right: bevy::ui::Val::Px(0.),
                    ..Default::default()
                },
                // Render overlay on top of everything
                GlobalZIndex(FPS_OVERLAY_ZINDEX),
            ))
            .with_children(|p| {
                p.spawn((
                    Text::new("FPS: "),
                    overlay_config.text_config.clone(),
                    TextColor(overlay_config.text_color),
                    FpsText,
                ))
                .with_child((TextSpan::default(), overlay_config.text_config.clone()));
            });
    }

    fn update_text(
        diagnostic: Res<DiagnosticsStore>,
        query: Query<Entity, With<FpsText>>,
        mut writer: TextUiWriter,
        time: Res<Time>,
        config: Res<FpsOverlayConfig>,
        mut time_since_rerender: Local<Duration>,
    ) {
        *time_since_rerender += time.delta();
        if *time_since_rerender >= config.refresh_interval {
            *time_since_rerender = Duration::ZERO;
            for entity in &query {
                if let Some(fps) = diagnostic.get(&FrameTimeDiagnosticsPlugin::FPS) {
                    if let Some(value) = fps.smoothed() {
                        *writer.text(entity, 1) = format!("{value:.2}");
                    }
                }
            }
        }
    }

    fn customize_text(
        overlay_config: Res<FpsOverlayConfig>,
        query: Query<Entity, With<FpsText>>,
        mut writer: TextUiWriter,
    ) {
        for entity in &query {
            writer.for_each_font(entity, |mut font| {
                *font = overlay_config.text_config.clone();
            });
            writer.for_each_color(entity, |mut color| color.0 = overlay_config.text_color);
        }
    }

    fn toggle_display(
        overlay_config: Res<FpsOverlayConfig>,
        mut query: Query<&mut Visibility, With<FpsText>>,
    ) {
        for mut visibility in &mut query {
            visibility.set_if_neq(match overlay_config.enabled {
                true => Visibility::Visible,
                false => Visibility::Hidden,
            });
        }
    }
}

/// Generate disjoint query filters for the provided list of types.
///
/// Alternatively, you can generate the types in one step using the [`make_disjoint_markers`] macro.
///
/// # Usage
///
/// `disjoint!(A, B);`
///
/// # Example
/// ```
/// # use bevy::prelude::{App, Component, Update, Query, Transform};
/// use bevy_djqf::{Disjoint, disjoint};
///
/// #[derive(Component, Debug, Default)]
/// struct A;
///
/// #[derive(Component, Debug, Default)]
/// struct B;
///
/// disjoint!(A, B);
///
/// fn only_a(_query: Query<&mut Transform, <A as Disjoint>::Only>) {}
///
/// fn except_a(_query: Query<&mut Transform, <A as Disjoint>::Other>) {}
///
/// # App::new().add_systems(Update, (only_a, except_a));
/// ```
#[macro_export]
macro_rules! disjoint {
    // entry point: 2+ types
    ( $current:ty, $( $rest:ty ),* ) => {
        $crate::disjoint!(@for [] $current [ $( $rest , )* ]);
    };

    // entry point: 1 type
    ( $current:ty ) => {
        const _: () = panic!("You must provide at least two types");
    };

    // entry point: 0 types
    () => {
        const _: () = panic!("You must provide at least two types");
    };

    // 2+ remaining
    (@for [ $( $consumed:ty , )* ] $current:ty [ $next:ty , $( $later:ty , )* ]) => {
        $crate::disjoint!(@imp [ $( $consumed , )* ] $current [ $next , $( $later , )* ]);
        $crate::disjoint!(@for [ $( $consumed , )* $current , ] $next [ $( $later , )* ]);
    };

    // 1 remaining
    (@for [ $( $consumed:ty , )* ] $current:ty [ $next:ty ]) => {
        $crate::disjoint!(@imp [ $( $consumed , )* ] $current [ $next ]);
        $crate::disjoint!(@for [ $( $consumed , )* $current , ] $next []);
    };

    // 0 remaining
    (@for [ $( $consumed:ty , )* ] $current:ty []) => {
        $crate::disjoint!(@imp [ $( $consumed , )* ] $current []);
    };

    (@imp [ $( $before:ty , )* ] $current:ty [ $( $after:ty , )* ]) => {
        impl $crate::Disjoint for $current {

            type Any = bevy::ecs::query::Or<(
                $(bevy::ecs::query::With<$before> , )*
                bevy::ecs::query::With<$current> ,
                $(bevy::ecs::query::With<$after> , )*
            )>;

            type Other = (
                bevy::ecs::query::Without<$current> ,
                bevy::ecs::query::Or<(
                    $(bevy::ecs::query::With<$before> , )*
                    $(bevy::ecs::query::With<$after> , )*
                )>
            );

            type Only = (
                $(bevy::ecs::query::Without<$before> , )*
                bevy::ecs::query::With<$current> ,
                $(bevy::ecs::query::Without<$after> , )*
            );

        }
    };

    ( $($invalid_input:tt)* ) => {
        const _: () = panic!(
            concat!(
                "Invalid input `",
                stringify!($($invalid_input)*),
                "` to macro `disjoint!`. Use the form `disjoint!(A, B)`"
            )
        );
    };
}

/// A trait for disjoint queries. The `Any`, `Other`, and `Only` associated types are generated by the [`disjoint!`] macro.
///
/// These can be used in queries like `Query<&mut Transform, <A as Disjoint>::Only>`.
pub trait Disjoint {
    /// Any entities for this "enum".
    type Any;
    /// Entities that do not have this specific "variant".
    type Other;
    /// Entities that only have this specific "variant".
    type Only;
}

/// Type aliases for querying a single entity for one of the items
pub type OnlySingle<'w, QueryData, Variant, OtherFilters = ()> =
    Single<'w, QueryData, (<Variant as Disjoint>::Only, OtherFilters)>;

/// Type aliases for querying many entities for one of the items
pub type OnlyQuery<'w, 's, QueryData, Variant, OtherFilters = ()> =
    Query<'w, 's, QueryData, (<Variant as Disjoint>::Only, OtherFilters)>;

/// Generate marker types for disjoint query filters for the provided list of names.
///
/// Alternatively, use existing types with the [`disjoint`] macro.
///
/// # Usage
///
/// `make_disjoint_markers!(type_template for A, B)` where `type_template` is the name of the macro.
///
/// # Example
/// ```
/// # use bevy::prelude::{App, Component, Update, Query, Transform};
/// use bevy_djqf::{make_disjoint_markers, Disjoint};
///
/// // Write a macro as a type_template for generating types
/// macro_rules! type_template {
///     ($Name:ident) => {
///         #[derive(Component, Debug, Default)]
///         struct $Name;
///     };
/// }
///
/// // Provide the macro and the list of type names you want to generate
/// make_disjoint_markers!(type_template for Player, FriendlyPlayer, EnemyPlayer, NonPlayerCharacter, FriendlyAi, EnemyAi);
///
/// fn player_only(
///     _player_only: Query<&mut Transform, <Player as Disjoint>::Only>,
///     _others: Query<&mut Transform, <Player as Disjoint>::Other>,
/// ) {}
///
/// fn any(_query: Query<&mut Transform, <Player as Disjoint>::Any>) {}
///
/// # App::new().add_systems(Update, (player_only, any));
/// ```
#[macro_export]
macro_rules! make_disjoint_markers {
    ($type_template_macro:ident for $($Name:ident),*) => {
        $(
            $type_template_macro!($Name);
        )*

        $crate::disjoint!($($Name),*);
    };

    ( $($invalid_input:tt)* ) => {
        const _: () = panic!(
            concat!(
                "Invalid input `",
                stringify!($($invalid_input)*),
                "` to macro `make_disjoint_markers!`. Use the form `make_disjoint_markers!(type_template for A, B)` where `type_template` is the name of the macro"
            )
        );
    };
}

#[derive(Debug, Resource)]
pub struct Scale(f32);

impl Scale {
    fn factor(&self) -> f32 {
        (self.0 / 8.0).exp()
    }
}

#[derive(Resource, Clone, Copy)]
struct Extents {
    min_x: i32,
    max_x: i32,
    min_y: i32,
    max_y: i32,
}

impl From<Extents> for Rect {
    fn from(extents: Extents) -> Self {
        let Extents {
            min_x,
            max_x,
            min_y,
            max_y,
        } = extents;
        let offset_x = (min_x + max_x) as f32 / 2.;
        let offset_y = (min_y + max_y) as f32 / 2.;
        Rect::new(
            min_x as f32 + offset_x,
            min_y as f32 + offset_y,
            max_x as f32 + offset_x,
            max_y as f32 + offset_y,
        )
    }
}

#[derive(Resource)]
struct Once<T>(Option<T>);

impl<T> Once<T> {
    fn new(value: T) -> Self {
        Self(Some(value))
    }
    fn take(&mut self) -> Result<T> {
        self.0.take().ok_or("Already taken".into())
    }
}

struct MonitorImage {
    x: i32,
    y: i32,
    image: ImageBuffer<Rgba<u8>, Vec<u8>>,
}

// impl std::fmt::Debug for MonitorImage {
//     fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
//         let height = self.image.height();
//         let width = self.image.width();
//         #[derive(Debug)]
//         struct Image {
//             height: u32,
//             width: u32,
//         }
//         f.debug_struct("MonitorImage")
//             .field("x", &self.x)
//             .field("y", &self.y)
//             .field("image", &Image { height, width })
//             .finish()
//     }
// }

impl MonitorImage {
    fn new(monitor: Monitor) -> Result<Self> {
        let x = monitor.x()?;
        let y = monitor.y()?;
        let image = monitor.capture_image()?;
        Ok(Self { x, y, image })
    }
}

fn snapshot() -> Result<Vec<MonitorImage>> {
    let monitors = Monitor::all()?;

    let mut images = Vec::with_capacity(monitors.len());
    for monitor in monitors {
        let image = MonitorImage::new(monitor)?;
        images.push(image);
    }

    Ok(images)
}

trait RectExt {
    fn flip_up(self) -> Self;
    fn flip_right(self) -> Self;
    fn flip_down(self) -> Self;
    fn flip_left(self) -> Self;
    fn translate(self, offset: Vec2) -> Self;
    fn with_center(self, center: Vec2) -> Self;
    fn with_size(self, size: Vec2) -> Self;
    fn scale(self, factor: f32) -> Self;
    fn with_height(self, height: f32) -> Self;
    fn with_width(self, width: f32) -> Self;
    fn translate_x(self, offset: f32) -> Self;
    fn translate_y(self, offset: f32) -> Self;
    fn top_left(self) -> Vec2;
    fn top_center(self) -> Vec2;
    fn top_right(self) -> Vec2;
    fn center_right(self) -> Vec2;
    fn bottom_right(self) -> Vec2;
    fn bottom_center(self) -> Vec2;
    fn bottom_left(self) -> Vec2;
    fn center_left(self) -> Vec2;
    fn half_height(self) -> f32;
    fn half_width(self) -> f32;
    fn top(self) -> f32;
    fn bottom(self) -> f32;
    fn left(self) -> f32;
    fn right(self) -> f32;
}

impl RectExt for Rect {
    #[inline(always)]
    fn flip_up(self) -> Self {
        let offset = Vec2::new(0.0, self.height());
        self.translate(offset)
    }

    #[inline(always)]
    fn flip_right(self) -> Self {
        let offset = Vec2::new(self.width(), 0.0);
        self.translate(offset)
    }

    #[inline(always)]
    fn flip_down(self) -> Self {
        let offset = Vec2::new(0.0, -self.height());
        self.translate(offset)
    }

    #[inline(always)]
    fn flip_left(self) -> Self {
        let offset = Vec2::new(-self.width(), 0.0);
        self.translate(offset)
    }

    #[inline(always)]
    fn translate(mut self, offset: Vec2) -> Self {
        self.min += offset;
        self.max += offset;
        self
    }

    #[inline(always)]
    fn with_center(self, center: Vec2) -> Self {
        let offset = center - self.center();
        self.translate(offset)
    }

    #[inline(always)]
    fn with_size(self, size: Vec2) -> Self {
        Self::from_center_size(self.center(), size)
    }

    #[inline(always)]
    fn scale(mut self, factor: f32) -> Self {
        self.min *= factor;
        self.max *= factor;
        self
    }

    #[inline(always)]
    fn with_height(self, height: f32) -> Self {
        self.with_size(Vec2::new(self.width(), height))
    }

    #[inline(always)]
    fn with_width(self, width: f32) -> Self {
        self.with_size(Vec2::new(width, self.height()))
    }

    #[inline(always)]
    fn translate_x(self, offset: f32) -> Self {
        self.translate(Vec2::new(offset, 0.0))
    }

    #[inline(always)]
    fn translate_y(self, offset: f32) -> Self {
        self.translate(Vec2::new(0.0, offset))
    }

    #[inline(always)]
    fn top_left(self) -> Vec2 {
        Vec2::new(self.min.x, self.max.y)
    }

    #[inline(always)]
    fn top_center(self) -> Vec2 {
        Vec2::new(self.center().x, self.max.y)
    }

    #[inline(always)]
    fn top_right(self) -> Vec2 {
        self.max
    }

    #[inline(always)]
    fn center_right(self) -> Vec2 {
        Vec2::new(self.max.x, self.center().y)
    }

    #[inline(always)]
    fn bottom_right(self) -> Vec2 {
        Vec2::new(self.max.x, self.min.y)
    }

    #[inline(always)]
    fn bottom_center(self) -> Vec2 {
        Vec2::new(self.center().x, self.min.y)
    }

    #[inline(always)]
    fn bottom_left(self) -> Vec2 {
        self.min
    }

    #[inline(always)]
    fn center_left(self) -> Vec2 {
        Vec2::new(self.min.x, self.center().y)
    }

    #[inline(always)]
    fn half_height(self) -> f32 {
        self.half_size().y
    }

    #[inline(always)]
    fn half_width(self) -> f32 {
        self.half_size().x
    }

    #[inline(always)]
    fn top(self) -> f32 {
        self.top_right().y
    }

    #[inline(always)]
    fn bottom(self) -> f32 {
        self.bottom_left().y
    }

    #[inline(always)]
    fn left(self) -> f32 {
        self.bottom_left().x
    }

    #[inline(always)]
    fn right(self) -> f32 {
        self.top_right().x
    }
}

trait ValExt {
    fn px_or(self, fallback: f32) -> f32;
}

impl ValExt for Val {
    fn px_or(self, fallback: f32) -> f32 {
        match self {
            Self::Px(val) => val,
            _ => fallback,
        }
    }
}

mod components {
    use bevy::prelude::*;

    #[derive(Component, Default)]
    #[require(TargetViewportRect)]
    pub struct TargetWorldRect(pub Rect);

    #[derive(Component, Default)]
    pub struct TargetViewportRect(pub Rect);

    pub mod target_export {

        use crate::*;

        #[derive(Component, Default)]
        pub struct TargetExport;

        macro_rules! type_template {
            ($Name:ident) => {
                #[derive(Component, Debug, Default)]
                #[require(TargetExport)]
                pub struct $Name;
            };
        }
        make_disjoint_markers!(
            type_template for
            UiRoot,
            OuterLeft,
            OuterRight,
            OuterTop,
            OuterBottom,
            Target,
            HandleTopLeft,
            HandleTopCenter,
            HandleTopRight,
            HandleCenterRight,
            HandleBottomRight,
            HandleBottomCenter,
            HandleBottomLeft,
            HandleCenterLeft
        );
    }
}

mod systems {
    use bevy::{
        color::palettes::{
            css::{
                BLACK, BROWN, GOLDENROD, GREEN, LIGHT_BLUE, MAGENTA, ORANGE, PINK, RED, TEAL,
                WHITE, YELLOW,
            },
            tailwind::RED_400,
        },
        diagnostic::FrameCount,
        ecs::{query::QueryFilter, system::SystemParam},
        image::ImageSampler,
        input::mouse::{AccumulatedMouseMotion, AccumulatedMouseScroll},
        prelude::*,
        render::camera::{RenderTarget, ViewportConversionError},
        sprite::Anchor,
        window::{PrimaryWindow, SystemCursorIcon, WindowRef},
        winit::cursor::CursorIcon,
    };

    use crate::{
        Extents, MonitorImage, Once, OnlySingle, RectExt, Scale, ValExt,
        components::{target_export, *},
    };

    pub fn spawn_monitor_images(
        extents: Res<Extents>,
        mut images: ResMut<Once<Vec<MonitorImage>>>,
        mut commands: Commands,
        mut image_assets: ResMut<Assets<Image>>,
    ) -> Result {
        let images = images.take()?;

        for image in images {
            commands.spawn((
                Sprite {
                    image: image_assets.add(Image::from_dynamic(
                        image.image.into(),
                        true,
                        Default::default(),
                    )),
                    anchor: Anchor::TopLeft,
                    ..default()
                },
                Transform::from_xyz(
                    image.x as f32 - (extents.min_x as f32 + extents.max_x as f32) / 2.,
                    -(image.y as f32 - (extents.max_y as f32 + extents.min_y as f32) / 2.),
                    0.0,
                ),
            ));
        }

        Ok(())
    }

    pub fn spawn_camera(mut commands: Commands) {
        commands.spawn(Camera2d);
    }

    pub fn setup_window(mut commands: Commands, window: Single<Entity, With<Window>>) {
        let window = window.into_inner();
        commands
            .entity(window)
            .insert(CursorIcon::from(SystemCursorIcon::Default));
    }

    /// Window visibility is deliberately delayed
    pub fn make_window_visible(mut window: Single<&mut Window>, frames: Res<FrameCount>) {
        // The delay may be different for your app or system.
        if frames.0 == 1 {
            // At this point the gpu is ready to show the app so we can make the window visible.
            // Alternatively, you could toggle the visibility in Startup.
            // It will work, but it will have one white frame before it starts rendering
            window.visible = true;
        }
    }
    pub fn reset(
        key: Res<ButtonInput<KeyCode>>,
        mut scale: ResMut<Scale>,
        mut camera_transform: Single<&mut Transform, With<Camera>>,
        mut image_assets: ResMut<Assets<Image>>,
        sprites: Query<&Sprite>,
    ) {
        if key.pressed(KeyCode::Escape) {
            scale.0 = 0.0;
            camera_transform.translation = Vec3::ZERO;
            for sprite in sprites {
                sprite.image.get_mut(&mut image_assets).unwrap().sampler = ImageSampler::Default;
            }
        }
    }

    pub fn toggle_pixelation(
        mut nearest: Local<bool>,
        key: Res<ButtonInput<KeyCode>>,
        mut image_assets: ResMut<Assets<Image>>,
        sprites: Query<&Sprite>,
    ) {
        if key.just_pressed(KeyCode::Tab) {
            for sprite in sprites {
                let image = sprite.image.get_mut(&mut image_assets).unwrap();
                if *nearest {
                    // default from ImagePlugin is linear
                    image.sampler = ImageSampler::Default;
                } else {
                    image.sampler = ImageSampler::nearest();
                }
            }
            if *nearest {
                *nearest = false;
            } else {
                *nearest = true;
            }
        }
    }

    pub fn grab(
        mouse: Res<ButtonInput<MouseButton>>,
        movement: Res<AccumulatedMouseMotion>,
        scale: Res<Scale>,
        mut camera_transform: Single<&mut Transform, With<Camera>>,
    ) {
        let [x, y] = movement.delta.to_array();
        let adjustment = scale.factor();
        if mouse.pressed(MouseButton::Middle) {
            camera_transform.translation.x -= adjustment * x;
            camera_transform.translation.y += adjustment * y;
        }
    }

    #[derive(Resource, Default)]
    pub struct WorldCursor(Vec2);

    #[derive(Resource, Default)]
    pub struct ViewportCursor(Vec2);

    fn world_to_viewport(
        world_position: Vec2,
        camera: &Camera,
        camera_transform: &GlobalTransform,
    ) -> Result<Vec2, ViewportConversionError> {
        camera.world_to_viewport(camera_transform, world_position.extend(0.))
    }

    fn viewport_to_world(
        viewport_position: Vec2,
        camera: &Camera,
        camera_transform: &GlobalTransform,
    ) -> Result<Vec2, ViewportConversionError> {
        camera.viewport_to_world_2d(camera_transform, viewport_position)
    }

    #[derive(SystemParam)]
    pub struct ViewWorldConverter<'w, Filter: 'static + QueryFilter = ()> {
        camera: Single<'w, (&'static Camera, &'static GlobalTransform), Filter>,
    }

    impl<'w> ViewWorldConverter<'w> {
        fn world_to_viewport(&self, world_position: Vec2) -> Result<Vec2, ViewportConversionError> {
            world_to_viewport(world_position, self.camera.0, self.camera.1)
        }

        fn viewport_to_world(
            &self,
            viewport_position: Vec2,
        ) -> Result<Vec2, ViewportConversionError> {
            viewport_to_world(viewport_position, self.camera.0, self.camera.1)
        }

        fn camera_target(&self) -> &RenderTarget {
            &self.camera.0.target
        }
    }

    pub fn update_global_cursors(
        mut world_cursor: ResMut<WorldCursor>,
        mut viewport_cursor: ResMut<ViewportCursor>,
        // need to get window dimensions
        windows: Query<&Window>,
        primary_window: Single<&Window, With<PrimaryWindow>>,
        // query to get camera transform
        converter: ViewWorldConverter,
    ) {
        // get the window that the camera is displaying to (or the primary window)
        let window = if let RenderTarget::Window(WindowRef::Entity(id)) = converter.camera_target()
        {
            windows.get(*id).unwrap()
        } else {
            *primary_window
        };

        // check if the cursor is inside the window and get its position
        // then, ask bevy to convert into world coordinates, and truncate to discard Z
        if let Some(viewport_pos) = window.cursor_position()
            && let Ok(world_pos) = converter.viewport_to_world(viewport_pos)
        {
            viewport_cursor.0 = viewport_pos;
            world_cursor.0 = world_pos;
        }
    }

    #[derive(Default)]
    pub enum ResizeState {
        #[default]
        Idle,
        Drawing {
            w_start: Vec2,
        },
        Corner {
            w_opposite_corner: Vec2,
            v_cursor_handle_offset: Vec2,
        },
        Edge {
            w_opposite_corner_a: Vec2,
            w_opposite_corner_b: Vec2,
            v_cursor_handle_offset: Vec2,
        },
        Dragging {
            v_cursor_center_offset: Vec2,
        },
    }

    /// part of export tool
    /// TODO: break this up into many systems: start resize, while resize, finish resize, set cursor
    pub fn resize_target(
        mut gizmos: Gizmos,
        world_cursor: Res<WorldCursor>,
        viewport_cursor: Res<ViewportCursor>,
        mouse: Res<ButtonInput<MouseButton>>,
        target: Single<(&mut TargetWorldRect, &mut TargetViewportRect)>,
        mut state: Local<ResizeState>,
        window: Single<&mut CursorIcon, With<Window>>,
        converter: ViewWorldConverter,
    ) -> Result<()> {
        let mut cursor_icon = window.into_inner();

        let (wt, vt) = &mut target.into_inner();
        let wt = &mut wt.0;
        let vt = &mut vt.0;
        let wc = world_cursor.0;
        let vc = viewport_cursor.0;

        macro_rules! dbg_v {
            (circ / $color:expr, $expr:expr) => {{
                let it = $expr;
                gizmos.circle_2d(
                    Isometry2d::from_translation(converter.viewport_to_world(it)?),
                    10.,
                    $color,
                );
                it
            }};
            (rect / $color:expr, $expr:expr) => {{
                let it = $expr;
                gizmos.rect_2d(
                    Isometry2d::from_translation(converter.viewport_to_world(it)?),
                    Vec2::splat(20.),
                    $color,
                );
                it
            }};
        }
        macro_rules! dbg_w {
            (circ / $color:expr, $expr:expr) => {{
                let it = $expr;
                gizmos.circle_2d(Isometry2d::from_translation(it), 10., $color);
                it
            }};
            (rect / $color:expr, $expr:expr) => {{
                let it = $expr;
                gizmos.rect_2d(Isometry2d::from_translation(it), Vec2::splat(20.), $color);
                it
            }};
        }

        // TODO: this should be set in UI space not world space
        let v_handle_top_left = Rect::from_center_half_size(vt.bottom_left(), Vec2::splat(10.0));
        let v_handle_top_center =
            Rect::from_center_half_size(vt.bottom_center(), Vec2::splat(10.0));
        let v_handle_top_right = Rect::from_center_half_size(vt.bottom_right(), Vec2::splat(10.0));
        let v_handle_center_right =
            Rect::from_center_half_size(vt.center_right(), Vec2::splat(10.0));
        let v_handle_bottom_right = Rect::from_center_half_size(vt.top_right(), Vec2::splat(10.0));
        let v_handle_bottom_center =
            Rect::from_center_half_size(vt.top_center(), Vec2::splat(10.0));
        let v_handle_bottom_left = Rect::from_center_half_size(vt.top_left(), Vec2::splat(10.0));
        let v_handle_center_left = Rect::from_center_half_size(vt.center_left(), Vec2::splat(10.0));

        let system_cursor_icon = match () {
            _ if v_handle_bottom_right.contains(vc) => {
                dbg_v!(circ / LIGHT_BLUE, v_handle_bottom_right.center());
                SystemCursorIcon::SeResize
            }
            _ if v_handle_top_right.contains(vc) => {
                dbg_v!(circ / GREEN, v_handle_top_right.center());
                SystemCursorIcon::NeResize
            }
            _ if v_handle_bottom_left.contains(vc) => {
                dbg_v!(circ / YELLOW, v_handle_bottom_left.center());
                SystemCursorIcon::SwResize
            }
            _ if v_handle_top_left.contains(vc) => {
                dbg_v!(circ / RED, v_handle_top_left.center());
                SystemCursorIcon::NwResize
            }
            _ if v_handle_bottom_center.contains(vc) => {
                dbg_v!(circ / BROWN, v_handle_bottom_center.center());
                SystemCursorIcon::SResize
            }
            _ if v_handle_center_right.contains(vc) => {
                dbg_v!(circ / MAGENTA, v_handle_center_right.center());
                SystemCursorIcon::EResize
            }
            _ if v_handle_top_center.contains(vc) => {
                dbg_v!(circ / ORANGE, v_handle_top_center.center());
                SystemCursorIcon::NResize
            }
            _ if v_handle_center_left.contains(vc) => {
                dbg_v!(circ / TEAL, v_handle_center_left.center());
                SystemCursorIcon::WResize
            }
            _ if vt.contains(vc) => {
                dbg_v!(circ / WHITE, vt.center());
                SystemCursorIcon::Move
            }
            _ => SystemCursorIcon::Default,
        };

        if let ResizeState::Idle = *state {
            *cursor_icon = system_cursor_icon.into();
        } else if let ResizeState::Corner { .. } = *state {
            *cursor_icon = system_cursor_icon.into();
        }

        let resize_state = match () {
            _ if v_handle_bottom_right.contains(vc) => ResizeState::Corner {
                w_opposite_corner: wt.top_left(),
                v_cursor_handle_offset: vc - v_handle_bottom_right.center(),
            },
            _ if v_handle_top_right.contains(vc) => ResizeState::Corner {
                w_opposite_corner: wt.bottom_left(),
                v_cursor_handle_offset: vc - v_handle_top_right.center(),
            },
            _ if v_handle_bottom_left.contains(vc) => ResizeState::Corner {
                w_opposite_corner: wt.top_right(),
                v_cursor_handle_offset: vc - v_handle_bottom_left.center(),
            },
            _ if v_handle_top_left.contains(vc) => ResizeState::Corner {
                w_opposite_corner: wt.bottom_right(),
                v_cursor_handle_offset: vc - v_handle_top_left.center(),
            },
            _ if v_handle_bottom_center.contains(vc) => ResizeState::Edge {
                w_opposite_corner_a: wt.top_left(),
                w_opposite_corner_b: wt.top_right(),
                v_cursor_handle_offset: vc - v_handle_bottom_center.center(),
            },
            _ if v_handle_center_right.contains(vc) => ResizeState::Edge {
                w_opposite_corner_a: wt.top_left(),
                w_opposite_corner_b: wt.bottom_left(),
                v_cursor_handle_offset: vc - v_handle_center_right.center(),
            },
            _ if v_handle_top_center.contains(vc) => ResizeState::Edge {
                w_opposite_corner_a: wt.bottom_left(),
                w_opposite_corner_b: wt.bottom_right(),
                v_cursor_handle_offset: vc - v_handle_top_center.center(),
            },
            _ if v_handle_center_left.contains(vc) => ResizeState::Edge {
                w_opposite_corner_a: wt.top_right(),
                w_opposite_corner_b: wt.bottom_right(),
                v_cursor_handle_offset: vc - v_handle_center_left.center(),
            },
            _ if vt.contains(vc) => ResizeState::Dragging {
                v_cursor_center_offset: vc - vt.center(),
            },
            _ => ResizeState::Drawing { w_start: wc },
        };

        match resize_state {
            ResizeState::Idle => {}
            ResizeState::Drawing { w_start } => {
                dbg_w!(rect / GOLDENROD, w_start);
            }
            ResizeState::Corner {
                w_opposite_corner, ..
            } => {
                dbg_w!(rect / GOLDENROD, w_opposite_corner);
            }
            ResizeState::Edge {
                w_opposite_corner_a,
                w_opposite_corner_b,
                ..
            } => {
                dbg_w!(rect / GOLDENROD, w_opposite_corner_a);
                dbg_w!(rect / GOLDENROD, w_opposite_corner_b);
            }
            ResizeState::Dragging { .. } => {}
        };

        if mouse.just_pressed(MouseButton::Left) {
            *state = resize_state;
        }

        if mouse.pressed(MouseButton::Left) {
            match *state {
                ResizeState::Drawing { w_start: start } => {
                    let end = converter.viewport_to_world(vc)?;
                    *wt = Rect::from_corners(start.round(), end.round());
                }
                ResizeState::Corner {
                    w_opposite_corner: opposite_corner,
                    v_cursor_handle_offset: cursor_handle_offset,
                } => {
                    let end = converter.viewport_to_world(vc - cursor_handle_offset)?;
                    *wt = Rect::from_corners(opposite_corner.round(), end.round());
                }
                ResizeState::Edge {
                    w_opposite_corner_a: opposite_corner_a,
                    w_opposite_corner_b: opposite_corner_b,
                    v_cursor_handle_offset: cursor_handle_offset,
                } => {
                    let end = converter.viewport_to_world(vc - cursor_handle_offset)?;
                    *wt = Rect::from_corners(opposite_corner_a.round(), opposite_corner_b.round())
                        .union_point(end.round());
                }
                ResizeState::Dragging {
                    v_cursor_center_offset: cursor_center_offset,
                } => {
                    let end = converter.viewport_to_world(vc - cursor_center_offset)?;
                    *wt = wt.with_center(end.round());
                    *wt = Rect::from_corners(wt.min.round(), wt.max.round());
                }
                ResizeState::Idle => {}
            }
        }

        if mouse.just_released(MouseButton::Left) {
            *state = ResizeState::Idle;
            *cursor_icon = SystemCursorIcon::Default.into();
        }

        Ok(())
    }

    /// part of export tool
    pub fn update_target_viewport_rect(
        target: Single<(Ref<TargetWorldRect>, &mut TargetViewportRect)>,
        camera: Single<(Ref<Camera>, Ref<GlobalTransform>)>,
    ) -> Result<()> {
        let (world_rect, mut viewport_rect) = target.into_inner();
        let (camera, camera_transform) = camera.into_inner();
        if !world_rect.is_changed() && !camera.is_changed() && !camera_transform.is_changed() {
            return Ok(());
        }
        let world_rect = world_rect.0;
        let min = camera.world_to_viewport(&camera_transform, world_rect.min.extend(0.0))?;
        let max = camera.world_to_viewport(&camera_transform, world_rect.max.extend(0.0))?;
        viewport_rect.0 = Rect::from_corners(min.round(), max.round());
        Ok(())
    }

    const BORDER_THICKNESS: f32 = 2.;

    /// part of export
    pub fn spawn_target_ui(mut commands: Commands) {
        const BACKGROUND_ALPHA: f32 = 0.8;

        let mut root = commands.spawn((
            target_export::UiRoot,
            Node {
                width: Val::Percent(100.),
                height: Val::Percent(100.),
                ..default()
            },
        ));
        root.with_children(|parent| {
            parent.spawn((
                target_export::OuterTop,
                Node {
                    position_type: PositionType::Absolute,
                    ..default()
                },
                BackgroundColor(BLACK.with_alpha(BACKGROUND_ALPHA).into()),
            ));
            parent.spawn((
                target_export::OuterBottom,
                Node {
                    position_type: PositionType::Absolute,
                    ..default()
                },
                BackgroundColor(BLACK.with_alpha(BACKGROUND_ALPHA).into()),
            ));
            parent.spawn((
                target_export::OuterLeft,
                Node {
                    position_type: PositionType::Absolute,
                    ..default()
                },
                BackgroundColor(BLACK.with_alpha(BACKGROUND_ALPHA).into()),
            ));
            parent.spawn((
                target_export::OuterRight,
                Node {
                    position_type: PositionType::Absolute,
                    ..default()
                },
                BackgroundColor(BLACK.with_alpha(BACKGROUND_ALPHA).into()),
            ));
            parent.spawn((
                TargetWorldRect::default(),
                target_export::Target,
                Node {
                    box_sizing: BoxSizing::ContentBox,
                    border: UiRect::all(Val::Px(BORDER_THICKNESS)),
                    ..default()
                },
                BorderColor(RED_400.into()),
            ));
            parent.spawn((
                target_export::HandleTopLeft,
                Node {
                    position_type: PositionType::Absolute,
                    border: UiRect::all(Val::Px(BORDER_THICKNESS * 2.)),
                    box_sizing: BoxSizing::ContentBox,
                    ..default()
                },
                BorderColor(RED_400.into()),
                BorderRadius::all(Val::Percent(50.)),
            ));
            parent.spawn((
                target_export::HandleTopCenter,
                Node {
                    position_type: PositionType::Absolute,
                    border: UiRect::all(Val::Px(BORDER_THICKNESS * 2.)),
                    box_sizing: BoxSizing::ContentBox,
                    ..default()
                },
                BorderColor(RED_400.into()),
                BorderRadius::all(Val::Percent(50.)),
            ));
            parent.spawn((
                target_export::HandleTopRight,
                Node {
                    position_type: PositionType::Absolute,
                    border: UiRect::all(Val::Px(BORDER_THICKNESS * 2.)),
                    box_sizing: BoxSizing::ContentBox,
                    ..default()
                },
                BorderColor(RED_400.into()),
                BorderRadius::all(Val::Percent(50.)),
            ));
            parent.spawn((
                target_export::HandleCenterRight,
                Node {
                    position_type: PositionType::Absolute,
                    border: UiRect::all(Val::Px(BORDER_THICKNESS * 2.)),
                    box_sizing: BoxSizing::ContentBox,
                    ..default()
                },
                BorderColor(RED_400.into()),
                BorderRadius::all(Val::Percent(50.)),
            ));
            parent.spawn((
                target_export::HandleBottomRight,
                Node {
                    position_type: PositionType::Absolute,
                    border: UiRect::all(Val::Px(BORDER_THICKNESS * 2.)),
                    box_sizing: BoxSizing::ContentBox,
                    ..default()
                },
                BorderColor(RED_400.into()),
                BorderRadius::all(Val::Percent(50.)),
            ));
            parent.spawn((
                target_export::HandleBottomCenter,
                Node {
                    position_type: PositionType::Absolute,
                    border: UiRect::all(Val::Px(BORDER_THICKNESS * 2.)),
                    box_sizing: BoxSizing::ContentBox,
                    ..default()
                },
                BorderColor(RED_400.into()),
                BorderRadius::all(Val::Percent(50.)),
            ));
            parent.spawn((
                target_export::HandleBottomLeft,
                Node {
                    position_type: PositionType::Absolute,
                    border: UiRect::all(Val::Px(BORDER_THICKNESS * 2.)),
                    box_sizing: BoxSizing::ContentBox,
                    ..default()
                },
                BorderColor(RED_400.into()),
                BorderRadius::all(Val::Percent(50.)),
            ));
            parent.spawn((
                target_export::HandleCenterLeft,
                Node {
                    position_type: PositionType::Absolute,
                    border: UiRect::all(Val::Px(BORDER_THICKNESS * 2.)),
                    box_sizing: BoxSizing::ContentBox,
                    ..default()
                },
                BorderColor(RED_400.into()),
                BorderRadius::all(Val::Percent(50.)),
            ));
        });
    }

    /// part of export tool
    pub fn update_target_ui(
        target: OnlySingle<(&mut Node, Ref<TargetViewportRect>), target_export::Target>,
        outer_top: OnlySingle<&mut Node, target_export::OuterTop>,
        outer_bottom: OnlySingle<&mut Node, target_export::OuterBottom>,
        outer_left: OnlySingle<&mut Node, target_export::OuterLeft>,
        outer_right: OnlySingle<&mut Node, target_export::OuterRight>,
        handle_top_left: OnlySingle<&mut Node, target_export::HandleTopLeft>,
        handle_top_center: OnlySingle<&mut Node, target_export::HandleTopCenter>,
        handle_top_right: OnlySingle<&mut Node, target_export::HandleTopRight>,
        handle_center_right: OnlySingle<&mut Node, target_export::HandleCenterRight>,
        handle_bottom_right: OnlySingle<&mut Node, target_export::HandleBottomRight>,
        handle_bottom_center: OnlySingle<&mut Node, target_export::HandleBottomCenter>,
        handle_bottom_left: OnlySingle<&mut Node, target_export::HandleBottomLeft>,
        handle_center_left: OnlySingle<&mut Node, target_export::HandleCenterLeft>,
        scale: Res<Scale>,
    ) {
        let (mut node, target_rect) = target.into_inner();
        if !target_rect.is_changed() && !scale.is_changed() {
            return;
        }
        let target_rect = target_rect.0;

        // TODO: bevy issues:
        // - BoxSizing::BorderBox is incorrect
        // - (rounding errors?) Border's inner box is smaller than Background's inner box, when sufficiently zoomed

        node.width = Val::Px(target_rect.width());
        node.height = Val::Px(target_rect.height());
        node.top = Val::Px(target_rect.bottom_left().y - node.border.top.px_or(0.));
        node.left = Val::Px(target_rect.bottom_left().x - node.border.left.px_or(0.));

        // TODO: detect actual window corners instead
        let mut node = outer_top.into_inner();
        node.width = Val::Percent(100.);
        node.height = Val::Px(10000.);
        node.top = Val::Px(target_rect.top());
        node.left = Val::Px(0.);

        let mut node = outer_bottom.into_inner();
        node.width = Val::Percent(100.);
        node.height = Val::Px(10000.);
        node.top = Val::Px(-10000. + target_rect.bottom());
        node.left = Val::Px(0.);

        let mut node = outer_left.into_inner();
        node.width = Val::Px(10000.);
        node.height = Val::Px(target_rect.height());
        node.top = Val::Px(target_rect.bottom());
        node.left = Val::Px(-10000. + target_rect.left());

        let mut node = outer_right.into_inner();
        node.width = Val::Px(10000.);
        node.height = Val::Px(target_rect.height());
        node.top = Val::Px(target_rect.bottom());
        node.left = Val::Px(target_rect.right().floor());

        let mut node = handle_top_left.into_inner();
        node.top = Val::Px(target_rect.top_left().y - node.border.top.px_or(0.));
        node.left = Val::Px(target_rect.top_left().x - node.border.left.px_or(0.));

        let mut node = handle_top_center.into_inner();
        node.top = Val::Px(target_rect.top_center().y - node.border.top.px_or(0.));
        node.left = Val::Px(target_rect.top_center().x - node.border.left.px_or(0.));

        let mut node = handle_top_right.into_inner();
        node.top = Val::Px(target_rect.top_right().y - node.border.top.px_or(0.));
        node.left = Val::Px(target_rect.top_right().x - node.border.left.px_or(0.));

        let mut node = handle_center_right.into_inner();
        node.top = Val::Px(target_rect.center_right().y - node.border.top.px_or(0.));
        node.left = Val::Px(target_rect.center_right().x - node.border.left.px_or(0.));

        let mut node = handle_bottom_right.into_inner();
        node.top = Val::Px(target_rect.bottom_right().y - node.border.top.px_or(0.));
        node.left = Val::Px(target_rect.bottom_right().x - node.border.left.px_or(0.));

        let mut node = handle_bottom_center.into_inner();
        node.top = Val::Px(target_rect.bottom_center().y - node.border.top.px_or(0.));
        node.left = Val::Px(target_rect.bottom_center().x - node.border.left.px_or(0.));

        let mut node = handle_bottom_left.into_inner();
        node.top = Val::Px(target_rect.bottom_left().y - node.border.top.px_or(0.));
        node.left = Val::Px(target_rect.bottom_left().x - node.border.left.px_or(0.));

        let mut node = handle_center_left.into_inner();
        node.top = Val::Px(target_rect.center_left().y - node.border.top.px_or(0.));
        node.left = Val::Px(target_rect.center_left().x - node.border.left.px_or(0.));
    }

    pub fn zoom(
        scroll: Res<AccumulatedMouseScroll>,
        mut camera_projection: Single<&mut Projection, With<Camera>>,
        mut scale: ResMut<Scale>,
    ) {
        scale.0 -= scroll.delta.y;

        let Projection::Orthographic(ref mut orthographic_projection) = **camera_projection else {
            return;
        };

        orthographic_projection.scale = scale.factor();
    }

    trait HandleExt<A: Asset> {
        fn get<'a>(&self, assets: &'a Assets<A>) -> Option<&'a A>;
        fn get_mut<'a>(&self, assets: &'a mut Assets<A>) -> Option<&'a mut A>;
    }

    impl<A: Asset> HandleExt<A> for Handle<A> {
        fn get<'a>(&self, assets: &'a Assets<A>) -> Option<&'a A> {
            assets.get(self)
        }

        fn get_mut<'a>(&self, assets: &'a mut Assets<A>) -> Option<&'a mut A> {
            assets.get_mut(self)
        }
    }
}
